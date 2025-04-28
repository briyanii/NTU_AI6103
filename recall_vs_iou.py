import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils as U
from dataset import VOCDataset
from dataloader import get_dataloader
from models import FasterRCNN

parser = argparse.ArgumentParser()
parser.add_argument('--n_classes', type=int, default=20)
# parser.add_argument('--iou_threshold', type=float, default=.5)
parser.add_argument('--n_proposals', type=int, default=300)
parser.add_argument('--year', type=int, default=2007)
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--model_weight', type=str, default='./outputs/checkpoint_step4_80000.pt')

args = parser.parse_args()
n_classes = args.n_classes
n_proposals = args.n_proposals
# iou_th = args.iou_threshold
year = args.year
split = args.split
model_weight = args.model_weight

iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
# iou_thresholds = np.arange(0.5, 1.02, 0.02)  # (start=0.5, stop=1.02, step=0.02)
# iou_thresholds = [round(x, 2) for x in iou_thresholds]



device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight_path = model_weight
state = torch.load(weight_path, map_location='cpu')
model = FasterRCNN()
model.load_state_dict(state['model'])
model = model.to(device)
model.eval()

dataset = VOCDataset(year, split)
dataloader, _ = get_dataloader(
    dataset,
    num_samples=len(dataset),
    skip=0,
    normalize=True,
    augment=False,
    shuffle=False,
    batch_size=1,
    drop_last=False,
    num_workers=8,
    prefetch_factor=4
)

recall_per_thr = []
total_matched_per_thr = [0 for _ in iou_thresholds]
total_gt = 0

with torch.no_grad():
    for step, batch in enumerate(dataloader):
        imgs, _ = batch['x']
        imgs = list(map(lambda x: x.to(device), imgs))
        outputs, _ = model((imgs, None))

        cls_pred = outputs['det_cls_pred'].cpu()
        bbox_pred = outputs['det_bbox_pred'].cpu()
        score = outputs['det_score'].cpu()

        bbox_gt = torch.cat(batch['y']['bboxes'])
        cls_gt = torch.cat(batch['y']['class_ids'])

        im_w = batch['y']['width'][0]
        im_h = batch['y']['height'][0]

        
        recall_per_thr = []
        # for thr in iou_thresholds:
        gt_num = 0 # record number of gt
        #     matched_gt = 0 # record
        for idx_thr, thr in enumerate(iou_thresholds):
            matched_gt = 0
            # gt_num = 0
            for cls_id in range(1,n_classes+1):   
                # predicitions for class
                _cls_pred = cls_pred[cls_pred == cls_id]
                _bbox_pred = bbox_pred[cls_pred == cls_id]
                _score = score[cls_pred == cls_id]

                # take top n
                indices = _score.argsort(descending=True)[:n_proposals]
                _cls_pred = _cls_pred[indices]
                _bbox_pred = _bbox_pred[indices]
                _score = _score[indices]

                # ground truth for class
                _bbox_gt = bbox_gt[cls_gt == cls_id]
                _cls_gt = cls_gt[cls_gt == cls_id]

                # compute ious
                ious = U.compute_iou(_bbox_pred, _bbox_gt)
                N,M = ious.shape

                if N == 0 or  M == 0:
                    continue

                gt_num += M
                max_iou_per_gt, pred_idx_per_gt = ious.max(dim=0)

                matched_bbox_index = set()
                for i in range(M):
                    max_iou = max_iou_per_gt[i]
                    pred_idx = pred_idx_per_gt[i]
                    if max_iou >= thr and pred_idx not in matched_bbox_index:
                        matched_gt += 1
                        matched_bbox_index.add(pred_idx)

            total_matched_per_thr[idx_thr] += matched_gt
        total_gt += (gt_num / len(iou_thresholds))

for matched in total_matched_per_thr:
    recall = matched / total_gt
    recall_per_thr.append(recall)
# plot and save recall-iou curve
os.makedirs(f'./outputs/recall_vs_iou_curves/{year}_{split}', exist_ok=True)
plt.figure()
plt.plot(iou_thresholds, recall_per_thr)
plt.xlabel('IoU')
plt.ylabel('Recall')
plt.title(f'Recall vs IoU curve')
plt.grid(True)
plt.savefig(f'./outputs/recall_vs_iou_curves/{year}_{split}/recall_vs_iou_curve.png')

