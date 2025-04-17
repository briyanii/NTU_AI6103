import torch
from datetime import datetime
import pickle
import os
import glob
import sys
from trainer import Trainer
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

class TrainerStep4(Trainer):
    def after_step(self, step, loss):
        if loss.isnan():
            path = checkpoint_filename_template.format(step=step)
            self.save_state(path+'.nan')
            print("step {} Loss is NaN. Check for issues?".format(step))
            sys.exit(1)

        s = step+1
        now = datetime.now()
        now = now.ctime()
        print("{} - step {} of {} | loss = {:.3f}".format(now, s, self.training_steps, loss.item()))
        sys.stdout.flush()

        if (s % 10000 == 0):
            path = checkpoint_filename_template.format(step=step)
            self.save_state(path)
            print("Saved", path)

    def after_train(self):
        path = checkpoint_filename_template.format(step=self.training_steps)
        self.save_state(path)
        print("Saved", path)

def get_latest_checkpoint_path(template):
    glob_path = template.format(step="*")
    highest_step = -1

    for filepath in glob.glob(glob_path):
        filename = os.path.split(filepath)[1]
        step = filename.split('_')[-1]
        step = step.split('.pt')[0]
        step = int(step)
        highest_step = max(highest_step, int(step))

    if highest_step == -1:
        return
    return template.format(step=highest_step)

def load_latest_checkpoint(trainer):
    path = get_latest_checkpoint_path(checkpoint_filename_template)
    if path:
        trainer.load_state(path)

def load_state_dict(model):
    path = get_latest_checkpoint_path(init_weight_path)
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model'])

dataloader_seed=13223
_dataset = (2007, 'trainval')

if __name__ == '__main__':
    dataset = VOCDataset(*_dataset)

    def lr_lambda(step):
        if step < config['rpn_step0']:
            return config['rpn_lr_0']
        else:
            return config['rpn_lr_1']

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

    trainer = TrainerStep4(
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
    )

    load_latest_checkpoint(trainer)
    trainer.train()
