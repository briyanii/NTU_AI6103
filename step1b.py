from config import (
    get_rpn_cfg,
    get_anchor_cfg,
    roi_proposal_path,
    checkpoint_filename_template_1 as checkpoint_filename_template,
)
import utils as U
from torchvision.ops import nms
from models import RPN
from dataset import VOCDataset
import torch
import glob
import pickle
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    state = torch.load(path, map_location='cpu')
    return state['model']


if __name__ == '__main__':
    with torch.no_grad():
        model = RPN(get_rpn_cfg(), get_anchor_cfg())
        model = model.to(device)

        dataset = VOCDataset(2007, 'trainval')
        state_dict = load_latest_checkpoint()
        model.load_state_dict(state_dict)

        proposals = []
        for i, item in enumerate(dataset):
            img = item['image'].unsqueeze(0).to(device)
            i = item['index']
            w = item['width']
            h = item['height']
            outputs = model(img)
            anchors = outputs.anchors
            roi = outputs.roi_proposals
            cls_softmax = outputs.cls_softmax
            score = cls_softmax[0,:,0] # 0 = fg, 1 = bg

            # drop cross boundary objects for training process
            kept = U.drop_cross_boundary_boxes(anchors, w, h).to(roi.device)
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
            proposals.append(roi)

        with open(roi_proposal_path, 'wb') as fp:
            pickle.dump(proposals, fp)

