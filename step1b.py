from config import (
    roi_proposal_path,
    checkpoint_filename_template_1 as checkpoint_filename_template,
)
from dataloader import get_dataloader
import utils as U
from torchvision.ops import nms
from models import FasterRCNN
from dataset import VOCDataset
import torch
import glob
import pickle
import os

def load_latest_checkpoint(model):
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
    model.load_state_dict(state['model'])
    return model


if __name__ == '__main__':
    with torch.no_grad():
        model = FasterRCNN()
        dataset = VOCDataset(2007, 'trainval')
        n = 1#len(dataset)
        dataloader, _= get_dataloader(dataset,
            num_samples=n,
            seed=None,
            batch_size=1,
            drop_last=True,
            skip=0,
            augment=False,
            normalize=True,
            shuffle=False
        )

        model = load_latest_checkpoint(model)

        proposals = []
        for i in range(n):
            b = next(dataloader)
            output, _ = model(inputs, None)
            proposals.append(output['rpn_roi'])

        with open(roi_proposal_path + "_test", 'wb') as fp:
            pickle.dump(proposals, fp)

