from trainer import Trainer
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

class TrainerStep3(Trainer):
    def after_step(self, step, loss):
        if loss.isnan():
            path = checkpoint_filename_template.format(step=step)
            self.save_state(path + '.nan')
            print("step {} Loss is NaN. Check for issues?".format(step))
            sys.exit(1)

        s = step+1
        #if (s < 10) or (s % 100 == 0):
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
    path = template.format(step=highest_step)
    return path

def load_latest_checkpoint(trainer):
    path = get_latest_checkpoint_path(checkpoint_filename_template)
    if path:
        trainer.load_state(path)

def load_state_dict(model):
    path = get_latest_checkpoint_path(init_weight_path)
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model'])

dataloader_seed=54321
_dataset = (2007, 'trainval')

if __name__ == '__main__':
    def lr_lambda(step):
        if step < config['rpn_step0']:
            return config['rpn_lr_0']
        else:
            return config['rpn_lr_1']

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

    trainer = TrainerStep3(
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
    )

    load_latest_checkpoint(trainer)
    trainer.train()
    print("DONE")
