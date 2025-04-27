import os
import sys
import argparse
import pickle
import torch
import utils as U
from dataset import VOCDataset
from dataloader import get_dataloader
from models import FasterRCNN
from config import (
    checkpoint_filename_template_4,
    checkpoint_filename_template_3,
    checkpoint_filename_template_2,
    checkpoint_filename_template_1,
)
from comet_ml import Experiment

parser = argparse.ArgumentParser()
parser.add_argument('--n_classes', type=int, default=20)
parser.add_argument('--iou_threshold', type=float, default=.5)
parser.add_argument('--n_proposals', type=int, default=300)
parser.add_argument('--year', type=int, default=2007)
parser.add_argument('--split', type=str, default='test')

args = parser.parse_args()
n_classes = args.n_classes
n_proposals = args.n_proposals
iou_th = args.iou_threshold
year = args.year
split = args.split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight_path = checkpoint_filename_template_4.format(step=80000)
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
    num_workers=2,
    prefetch_factor=2
)


tps = {}
fps = {}
n_gts = {}

for cls_id in range(1, n_classes+1):
    tps[cls_id] = []
    fps[cls_id] = []
    n_gts[cls_id] = 0

experiment = Experiment(
    project_name="faster-rcnn",
    workspace="ai6103",
    auto_param_logging=False,
    auto_metric_logging=False,
    auto_output_logging=False,
    auto_log_co2=False,
    log_code=False,
    disabled=os.getenv('COMET_DISABLED', 'False') == 'True',
)
experiment.add_tag("eval")
experiment.log_parameters({
    'n_proposals': n_proposals,
    'iou_threshold': iou_th,
    'eval_dataset': '{} {}'.format(year, split),
    'n_images': len(dataset),
    'weight_path': weight_path,
})

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

        # class 0 = background
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

            n_gts[cls_id] += M
            max_iou_per_gt, pred_idx_per_gt = ious.max(dim=0)
            max_iou_per_pred, gt_idx_per_pred = ious.max(dim=1)
            matched = set()
            for i in range(N):
                max_iou = max_iou_per_pred[i]
                gt_idx = gt_idx_per_pred[i]
                if max_iou >= iou_th and gt_idx not in matched:
                    tps[cls_id].append(1)
                    fps[cls_id].append(0)
                    matched.add(gt_idx)
                else:
                    tps[cls_id].append(0)
                    fps[cls_id].append(1)
        experiment.log_text(f'{n_gts}', step=step)

APs = []
for cls_id in range(1, n_classes+1):
    _tps = torch.Tensor(tps[cls_id])
    _fps = torch.Tensor(fps[cls_id])
    _tps = torch.cumsum(_tps, 0)
    _fps = torch.cumsum(_fps, 0)
    precision = _tps / (_tps + _fps)
    recall = _tps / n_gts[cls_id]
    '''
    interp_precision = torch.zeros(N)
    for i in range(N):
        r = recall[i]
        idx = torch.where(recall >= r)
        max_precision_at_recall = precision[idx].max()
        interp_precision[idx] = max_precision_at_recall
    print(interp_precision)
    '''
    # VOC 2008+ interpolation approach
    recall_thresholds = recall.unique()
    AP = 0
    for i in range(1, recall_thresholds.shape[0]):
        r_n0 = recall_thresholds[i-1]
        r_n1 = recall_thresholds[i]
        idx = torch.where(recall >= r_n1)
        max_precision_at_recall = precision[idx].max()
        area = (r_n1 - r_n0) * max_precision_at_recall
        AP += area
    experiment.log_metric('AP', AP, step=cls_id)
    APs.append(AP)

mAP = torch.Tensor(APs).mean()
print(APs)
print(mAP)
experiment.log_metric('mAP', mAP.item())

eval_output = {
    'tps': tps,
    'fps': fps,
    'n_gts': n_gts,
    'APs': APs,
    'mAP': mAP,
}

fname = 'outputs/results.pkl'
with open(fname, 'wb') as fp:
    pickle.dump(eval_output, fp)
experiment.log_asset(fname)

