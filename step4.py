import torch
from datetime import datetime
import pickle
import os
import glob
import sys
from trainer import Trainer, get_latest_checkpoint
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from config import (
    checkpoint_filename_template_4 as checkpoint_filename_template,
    checkpoint_filename_template_3 as init_weight_path,
)
from config import config
from models import FasterRCNN, FasterRCNN_FastLoss
from dataset import VOCDataset

def load_state_dict(model):
    path = get_latest_checkpoint(init_weight_path)
    if path is None:
        return
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model'])

dataloader_seed=13223
_dataset = (2007, 'trainval')

if __name__ == '__main__':
    dataset = VOCDataset(*_dataset)

    def lr_lambda(step):
        if step < config['rpn_step0']:
            return 1
        else:
            return config['rpn_lr_1']/config['rpn_lr_0']

    model = FasterRCNN()
    # load state dict from previous step
    load_state_dict(model)

    # freeze backbone
    for n,m in model.features.named_parameters():
        m.requires_grad = False
    # freeze RPN
    for n,m in model.rpn_layer.named_parameters():
        m.requires_grad = False
    # train detection head only

    optimizer = SGD(model.parameters(), lr=config['rpn_lr_0'], weight_decay=config['sgd_decay'], momentum=config['sgd_momentum'])
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = FasterRCNN_FastLoss()

    trainer = Trainer(
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        steps=config['rpn_step0'] + config['rpn_step1'],
        accumulation_steps=2,
        dataloader_seed=dataloader_seed,
        batch_size=1,
        drop_last=True,
        augment_th=.5,
        normalize=True,
        augment=True,
        shuffle=True,
        num_workers=2,
        prefetch_factor=2,
        checkpoint_template=checkpoint_filename_template,
        experiment_tags=['step4']
    )

    load_latest_checkpoint(trainer)
    trainer.train()
