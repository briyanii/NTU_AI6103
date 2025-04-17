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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model'])


if __name__ == '__main__':
    with torch.no_grad():
        dataset = VOCDataset(2007, 'trainval')
        print('dataset ready')

        n = len(dataset)
        dataloader, _= get_dataloader(dataset,
            num_samples=n,
            seed=None,
            batch_size=1,
            drop_last=True,
            skip=0,
            augment=False,
            normalize=True,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2
        )
        print('dataloader ready')

        model = FasterRCNN()
        model = model.to(device)
        load_latest_checkpoint(model)
        model.eval()
        print('modedl ready')

        proposals = []
        i = 0
        for b in dataloader:
            images, _ = b['x']
            images[0] = images[0].to(device)
            inputs = (images, None)
            targets = None
            output, _ = model(inputs, targets)
            proposals.append(output['rpn_roi'].cpu())
            print(f'{i} of {n} done')
            i += 1

        print('proposals generated')
        with open(roi_proposal_path, 'wb') as fp:
            pickle.dump(proposals, fp)
        print('saved to', roi_proposal_path)

