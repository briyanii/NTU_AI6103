from trainer import Trainer
from datetime import datetime
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from config import (
    checkpoint_filename_template_1 as checkpoint_filename_template
)
from config import config
from models import FasterRCNN, FasterRCNN_RPNLoss
from dataset import VOCDataset
import torch
import sys
import glob
import os

dataloader_seed=54321
_dataset = (2007, 'trainval')

if __name__ == '__main__':
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
    for n,m in model.detection_layer.named_parameters():
        m.requires_grad = False

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
        dataloader_seed=54321,
        accumulation_steps=1,
        num_workers=2,
        prefetch_factor=2,
        checkpoint_template=checkpoint_filename_template,
        experiment_tags=['step1a']
    )

    trainer.load_latest_checkpoint()
    trainer.train()
    print("DONE")
