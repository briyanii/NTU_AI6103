from trainer import Trainer
from datetime import datetime
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import MaxPool2d
from datatypes import Rpn_cfg, Anchor_cfg
from models import RPN, RPNLoss_v2
from dataset import VOCDataset
import torch
import glob
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint_filename_template = './outputs/checkpoint_step1_{step}.pt'

class TrainerStep1(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def before_step(self, step):
        pass

    def after_step(self, step, loss):
        if loss.isnan():
            path = checkpoint_filename_template.format(step=step)
            self.save_state(path)
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


stride_len = 16
scales = [128,256,512]
areas = list(map(lambda x: x**2, scales))
ratios = [(1,1), (1,2), (2,1)]
k = len(areas) * len(ratios)
cfg_1 = Rpn_cfg(3, 512, k, 0.0, 0.01)
cfg_2 = Anchor_cfg(areas, ratios, stride_len, .3, .7, 256)

def lr_lambda(step):
    if step < 60000:
        return 0.001
    else:
        return 0.0001


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


def collate_fn(batch):
    item = batch[0]
    img = item['image'].unsqueeze(0).to(device)
    gt_bboxes = item['bboxes'].to(device)

    return {
        'x': img,
        'y': {
            # for loss
            'bboxes': gt_bboxes,
            # for dropping cross boundary anchors
            'width': item['width'],
            'height': item['height'],
        }
    }

total_steps = 60000+20000

if __name__ == '__main__':
    model = RPN(cfg_1, cfg_2)
    model = model.to(device)

    # paper says tofine tune entire thing, but freeze first few conv layers before max pool anyway
    for m in model.features.parameters():
        m.requires_grad = False
        if isinstance(m, MaxPool2d):
            break

    optimizer = SGD(model.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = RPNLoss_v2(10)
    dataset = VOCDataset('trainval')

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
    )

    load_latest_checkpoint(trainer)

    trainer.train()
    print("DONE")
