from config import (
    get_rpn_cfg,
    get_anchor_cfg,
    roi_proposal_path,
    checkpoint_filename_template_1 as checkpoint_filename_template,
)
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

        dataset = VOCDataset('trainval')
        state_dict = load_latest_checkpoint()
        model.load_state_dict(state_dict)

        proposals = []
        for i, item in enumerate(dataset):
            img = item['image'].unsqueeze(0)
            i = item['index']
            outputs = model(img)
            roi = outputs.roi_proposals
            indices = torch.full((roi.size(0), 1), i)
            roi = torch.hstack([indices, roi])
            proposals.append(roi)

        with open(roi_proposal_path, 'wb') as fp:
            pickle.dump(proposals, fp)

        print(roi_proposal_path)
        print(len(proposals))
