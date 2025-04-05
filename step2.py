import torch
from datetime import datetime
import sys
import utils as U
from trainer import Trainer
import torch.nn.functional as F
from torch.nn import MaxPool2d
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from config import (
    get_detection_cfg,
    get_anchor_cfg,
    roi_proposal_path,
    checkpoint_filename_template_2 as checkpoint_filename_template,
)
import pickle
from models import FastRCNN, MultiTaskLoss
from dataset import VOCDataset

class TrainerStep2(Trainer):
    def after_step(self, step, loss):
        if loss.isnan():
            path = checkpoint_filename_template.format(step=step)
            self.save_state(path+'.nan')
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


lr0 = 0.001
lr1 = 0.0001
steps0 = 30000
steps1 = 10000
total_steps = steps0 + steps1
R = 128
minibatch_N = 2
sample_size_per_img = R // minibatch_N
pos_th = .5
neg_lo = .1
neg_hi = .5
pos_ratio = .25
sgd_wd = 5e-4
sgd_momentum = 0.9
loss_lambda = 1.0
dataloader_seed=1249
_dataset = (2007, 'trainval')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = VOCDataset(*_dataset)

    with open(roi_proposal_path,'rb') as fp:
        roi_proposals = pickle.load(fp)

    def lr_lambda(step):
        if step < steps0:
            return lr0
        else:
            return lr1

    def prepare_one_item(item):
        img = item['image']
        idx = item['index']
        cls = item['class_ids']
        box = item['bboxes']
        rois = roi_proposals[idx][:, 1:]
        rois, gt_cls, gt_box = U.sample_rois(rois, cls, box, pos_th, neg_lo, neg_hi, sample_size_per_img, pos_ratio)
        return img, rois, gt_cls, gt_box

    def prepare_one_batch(batch):
        imgs = []
        _rois = []
        _gt_cls = []
        _gt_box = []
        h_max = 0
        w_max = 0
        for i, (img, rois, gt_cls, gt_box) in enumerate(map(prepare_one_item, batch)):
            rois = U.cat_val(rois, i)
            gt_cls = gt_cls.unsqueeze(1)
            gt_cls = U.cat_val(gt_cls, i)
            gt_box = U.cat_val(gt_box, i)
            h_max = max(h_max, img.size(1))
            w_max = max(w_max, img.size(2))
            imgs.append(img)
            _rois.append(rois)
            _gt_cls.append(gt_cls)
            _gt_box.append(gt_box)

        h_max = max([img.size(1) for img in imgs])
        w_max = max([img.size(2) for img in imgs])

        for i in range(minibatch_N):
            img = imgs[i]
            h,w = img.shape[1:]
            padded = F.pad(img, (0, w_max - w, 0, h_max - h))
            imgs[i] = padded

        imgs = torch.stack(imgs).to(device)
        rois = torch.cat(_rois).to(device)
        gt_box = torch.cat(_gt_box).to(device)
        gt_cls = torch.cat(_gt_cls).to(device)
        return {
            'x': (imgs, rois),
            'y': {
                'gt_box': gt_box,
                'gt_cls': gt_cls,
            }
        }

    model = FastRCNN(get_detection_cfg())
    model = model.to(device)
    '''
    # freeze up to conv3_1
    conv_block_id = 1
    for m in model.features.parameters():
        if isinstance(m, MaxPool2d):
            conv_block_id += 1
            if conv_block_id == 3:
                break
        m.requires_grad = False
    '''

    optimizer = SGD(model.parameters(), lr=lr0, weight_decay=sgd_wd, momentum=sgd_momentum)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = MultiTaskLoss(loss_lambda)

    trainer = TrainerStep2(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        steps=total_steps,
        dataset=dataset,
        dataloader_droplast=True,
        dataloader_batchsize=2,
        accumulation_steps=1,
        dataloader_samples=2*total_steps,
        dataloader_seed=54321,
        collate_fn=prepare_one_batch,
    )

    load_latest_checkpoint(trainer)
    trainer.train()
