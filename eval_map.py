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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight_path = checkpoint_filename_template_4.format(step=80000)
state = torch.load(weight_path, map_location='cpu')
model = FasterRCNN()
model.load_state_dict(state['model'])
model.eval()

dataset = VOCDataset(2007, 'trainval')
dataloader, _ = get_dataloader(
    dataset,
    num_samples=len(dataset),
    skip=0,
    normalize=False,
    augment=False,
    shuffle=False,
    batch_size=1,
    drop_last=False,
    num_workers=2,
    prefetch_factor=2
)

n_proposals = 300
iou_th = 0.5

with torch.no_grad():
    for batch in dataloader:

        imgs, _ = batch['x']
        imgs = list(map(lambda x: x.to(device), imgs))
        outputs, _ = model((imgs, None))

        cls_pred = outputs['det_cls_pred'].cpu()
        bbox_pred = outputs['det_bbox_pred'].cpu()
        score = outputs['det_score'].cpu()

        bbox_gt = torch.cat(batch['y']['bboxes'])
        cls_gt = torch.cat(batch['y']['class_ids'])

        MAP = torch.zeros(20)
        for cls_id in range(1,21):
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

            ious = U.compute_iou(_bbox_pred, _bbox_gt)
            N,M = ious.shape

            if N == 0 or  M == 0:
                continue

            max_iou_per_gt, pred_idx_per_gt = ious.max(dim=0)
            max_iou_per_pred, gt_idx_per_pred = ious.max(dim=1)
            matched = set()
            tps = torch.zeros(N)
            fps = torch.zeros(N)
            for i in range(N):
                max_iou = max_iou_per_pred[i]
                gt_idx = gt_idx_per_pred[i]
                if max_iou >= iou_th and gt_idx not in matched:
                    tps[i] = 1
                    matched.add(gt_idx)
                else:
                    fps[i] = 1
            tps = torch.cumsum(tps, 0)
            fps = torch.cumsum(fps, 0)
            precision = tps / (tps + fps)
            recall = tps / M
            '''
            interp_precision = torch.zeros(N)
            for i in range(N):
                r = recall[i]
                idx = torch.where(recall >= r)
                max_precision_at_recall = precision[idx].max()
                interp_precision[idx] = max_precision_at_recall
            print(interp_precision)
            '''
            recall_thresholds = recall.unique()
            AP = 0
            for i in range(1, recall_thresholds.shape[0]):
                r_n0 = recall_thresholds[i-1]
                r_n1 = recall_thresholds[i]
                idx = torch.where(recall >= r_n1)
                max_precision_at_recall = precision[idx].max()
                area = (r_n1 - r_n0) * max_precision_at_recall
                AP += area
            MAP[cls_id - 1] = AP

        print(MAP.mean())
        break


