from trainer import Trainer, get_latest_checkpoint
from datetime import datetime
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from config import (
    checkpoint_filename_template_3 as checkpoint_filename_template,
    checkpoint_filename_template_2 as init_weight_path,
)
from config import config
from models import FasterRCNN, FasterRCNN_RPNLoss
from dataset import VOCDataset
import torch
import sys
import glob
import os

def load_state_dict(model):
    path = get_latest_checkpoint(init_weight_path)
    if path is None:
        return
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model'])

dataloader_seed=54321
_dataset = (2007, 'trainval')

if __name__ == '__main__':
    def lr_lambda(step):
        if step < config['rpn_step0']:
            return 1
        else:
            return config['rpn_lr_1']/config['rpn_lr_0']

    model = FasterRCNN()
    load_state_dict(model)

    # freeze backbone
    for n,m in model.features.named_parameters():
        m.requires_grad = False
    # freeze detection head
    for n,m in model.detection_layer.named_parameters():
        m.requires_grad = False
    # train ONLY RPN head

    optimizer = SGD(model.parameters(), lr=config['rpn_lr_0'], weight_decay=config['sgd_decay'], momentum=config['sgd_momentum'])
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = FasterRCNN_RPNLoss()
    dataset = VOCDataset(*_dataset)

    trainer = Trainer(
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        steps=config['rpn_step0'] + config['rpn_step1'],
        batch_size=1,
        normalize=True,
        augment=True,
        shuffle=True,
        drop_last=False,
        dataloader_seed=dataloader_seed,
        accumulation_steps=1,
        num_workers=2,
        prefetch_factor=2,
        checkpoint_template=checkpoint_filename_template,
        experiment_tags=['step3']
    )

    load_latest_checkpoint(trainer)
    trainer.train()
    print("DONE")
