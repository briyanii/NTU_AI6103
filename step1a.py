from trainer import Trainer
from datetime import datetime
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import MaxPool2d
from config import (
    get_rpn_cfg,
    get_anchor_cfg,
    checkpoint_filename_template_1 as checkpoint_filename_template
)
from models import RPN, RPNLoss_v2
from dataset import VOCDataset
import torch
import sys
import glob
import os

class TrainerStep1(Trainer):
    def after_step(self, step, loss):
        if loss.isnan():
            path = checkpoint_filename_template.format(step=step)
            self.save_state(path + '.nan')
            print("step {} Loss is NaN. Check for issues?".format(step))
            sys.exit(1)

        s = step+1
        if (s < 10) or (s % 100 == 0):
            now = datetime.now()
            now = now.ctime()
            print("{} - step {} of {} | loss = {:.3f}".format(now, s, self.training_steps, loss.item()))

        if (s in [1, 1000, 2000]) or (s % 10000 == 0):
            path = checkpoint_filename_template.format(step=step)
            self.save_state(path)
            print("Saved", path)

    def after_train(self):
        path = checkpoint_filename_template.format(step=self.training_steps)
        self.save_state(path)
        print("Saved", path)

def load_latest_checkpoint(trainer):
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
    trainer.load_state(path)

lr_0 = 0.001
lr_1 = 0.0001
steps1 = 60000
steps2 = 20000
total_steps = steps1+steps2
loss_lambda = 10
sample_lo_th = .3
sample_hi_th = .7
sample_size = 256
sgd_wd = 5e-4
sgd_momentum = 0.9
dataloader_seed=54321
_dataset = (2007, 'trainval')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def lr_lambda(step):
        if step < steps1:
            return lr_0
        else:
            return lr_1

    def collate_fn(batch):
        item = batch[0]
        img = item['image'].unsqueeze(0).to(device)
        gt_bboxes = item['bboxes'].to(device)

        return {
            'x': img,
            'y': {
                'bboxes': gt_bboxes,
                'width': item['width'],
                'height': item['height'],
            }
        }

    model = RPN(get_rpn_cfg(), get_anchor_cfg())
    model = model.to(device)

    # freeze up to conv3_1
    conv_block_id = 1
    for m in model.features.parameters():
        if isinstance(m, MaxPool2d):
            conv_block_id += 1
            if conv_block_id == 3:
                break
        m.requires_grad = False

    optimizer = SGD(model.parameters(), lr=lr_0, weight_decay=sgd_wd, momentum=sgd_momentum)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = RPNLoss_v2(sample_lambda, sample_lo_th, sample_hi_th, sample_size)
    dataset = VOCDataset(*_dataset)

    trainer = TrainerStep1(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        steps=total_steps,
        dataset=dataset,
        dataloader_batchsize=1,
        dataloader_samples=total_steps,
        dataloader_seed=54321,
        collate_fn=collate_fn,
        accumulation_steps=1,
    )

    load_latest_checkpoint(trainer)

    trainer.train()
    print("DONE")
