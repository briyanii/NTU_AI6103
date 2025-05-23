import torch
from datetime import datetime
import pickle
import os
import glob
import sys
from trainer import Trainer
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from config import (
    roi_proposal_path,
    checkpoint_filename_template_2 as checkpoint_filename_template,
)
from config import config
from models import FasterRCNN, FasterRCNN_FastLoss
from dataset import VOCDataset


dataloader_seed=1249
_dataset = (2007, 'trainval')

if __name__ == '__main__':
    dataset = VOCDataset(*_dataset)

    def lr_lambda(step):
        if step < config['rpn_step0']:
            return 1
        else:
            return config['rpn_lr_1']/config['rpn_lr_0']

    model = FasterRCNN()

    # freeze up to conv3_1 (freeze conv1 and conv2)
    for n,m in model.features.named_parameters():
        n = n.split('.')[0]
        if n in ['0', '2', '5', '7']:
            m.requires_grad = False
    for n,m in model.rpn_layer.named_parameters():
        m.requires_grad = False

    optimizer = SGD(model.parameters(), lr=config['fast_lr_0'], weight_decay=config['sgd_decay'], momentum=config['sgd_momentum'])
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
        roi_proposal_path=roi_proposal_path,
        num_workers=2,
        prefetch_factor=2,
        checkpoint_template=checkpoint_filename_template,
        experiment_tags=['step2']
    )

    trainer.load_latest_checkpoint()
    trainer.train()
