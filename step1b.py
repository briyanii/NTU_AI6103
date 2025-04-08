from config import (
    roi_proposal_path,
    checkpoint_filename_template_1 as checkpoint_filename_template,
)
from dataloader import get_dataloader
import utils as U
from torchvision.ops import nms
from models import RPN
from dataset import VOCDataset
import torch
import glob
import pickle
import os

def load_latest_checkpoint():
    glob_path = checkpoint_filename_template.format(step="*")
    latest_checkpoint_path = None
    highest_step = -1

    for filepath in glob.glob(glob_path):
        filename = os.path.split(filepath)[1]
        step = filename.split('_')[-1]
        step = step.split('.pt')[0]
        step = int(step)
        highest_step = max(highest_step, int(step))

    if highest_step == -1:
        return

    path = checkpoint_filename_template.format(step=highest_step)
    state = torch.load(path)
    return state['model']


if __name__ == '__main__':
    with torch.no_grad():
        model = RPN()

        dataset = VOCDataset(2007, 'trainval')
        dataloader, _= get_dataloader(dataset,
            num_samples=len(dataset),
            seed=None,
            batch_size=1,
            drop_last=True,
            skip=0,
            augment=False,
            normalize=True,
            shuffle=False
        )

        state_dict = load_latest_checkpoint()
        model.load_state_dict(state_dict)

        proposals = []
        for i in range(len(dataset)):
            b = next(dataloader)
            img, _ = b['x']
            w = b['y']['width'][0]
            h = b['y']['height'][0]

            outputs = model(img)
            anchors = outputs.anchors
            roi = outputs.roi_proposals
            cls_softmax = outputs.cls_softmax
            score = cls_softmax[0, :, 1] # 1 = fg, 0 = bg

            # drop cross boundary anchors for training process
            kept = U.drop_cross_boundary_boxes(anchors, w, h)
            n_kept = kept.size(0) # ~60k
            roi = roi[kept]
            score = score[kept]

            # nms
            kept = nms(roi, score, .7)
            n_kept = kept.size(0) # ~2k
            score = score[kept]
            roi = roi[kept]

            indices = torch.full((n_kept, 1), i)
            roi = torch.hstack([indices, roi.cpu()])
            print('step {:15s}'.format(str(i+1)), roi.shape)
            proposals.append(roi)

        with open(roi_proposal_path, 'wb') as fp:
            pickle.dump(proposals, fp)

