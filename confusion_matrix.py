import argparse
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import utils as U
from dataset import VOCDataset
from dataloader import get_dataloader
from models import FasterRCNN

parser = argparse.ArgumentParser()
parser.add_argument('--n_classes', type=int, default=20)
parser.add_argument('--iou_threshold', type=float, default=.5)
parser.add_argument('--n_proposals', type=int, default=300)
parser.add_argument('--year', type=int, default=2007)
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--model_weight', type=str, default='./outputs/checkpoint_step4_80000-79999.pt')

args = parser.parse_args()
n_classes = args.n_classes
n_proposals = args.n_proposals
iou_th = args.iou_threshold
year = args.year
split = args.split
model_weight = args.model_weight

# First check if confmat has been saved
confmat_file = f'./outputs/confmat/{year}_{split}/confmat_no_bg.npy'

# Check if the confusion matrix file exists
if os.path.exists(confmat_file):
    # Load the confusion matrix from the file
    print("Loading confusion matrix from file...")
    confmat_no_bg = np.load(confmat_file)  # If using numpy to save/load

# if not saved, plot from scratch
else:
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

    # initialize conf matrix
    num_classes = n_classes
    confmat = torch.zeros((num_classes + 1, num_classes + 1), dtype=torch.int64)

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

            # filtering output bboxes
            # filtering 300 bboxes per image
            indices = score.argsort(descending=True)[:n_proposals]
            filter_cls_pred = cls_pred[indices]
            filter_bbox_pred = bbox_pred[indices]
            filter_score = score[indices]

            # compute ious
            ious = U.compute_iou(filter_bbox_pred, bbox_gt)
            N,M = ious.shape

            if N == 0 or  M == 0:
                continue

            max_iou_per_pred, gt_idx_per_pred = ious.max(dim=1)
            matched = set()
            for i in range(N):
                max_iou = max_iou_per_pred[i]
                gt_idx = gt_idx_per_pred[i]
                if max_iou >= iou_th and gt_idx not in matched:
                    # detection matched
                    gt_class = cls_gt[gt_idx]
                    pred_class = filter_cls_pred[i]
                    confmat[gt_class - 1, pred_class - 1] += 1


                    # tps[cls_id].append(1)
                    # fps[cls_id].append(0)
                    matched.add(gt_idx)
                else:
                    # tps[cls_id].append(0)
                    # fps[cls_id].append(1)
                    confmat[num_classes, pred_class - 1] += 1

            # After processing all predictions, add missed GTs (False Negatives)
            for gt_idx in range(len(bbox_gt)):
                if gt_idx not in matched:
                    gt_class = cls_gt[gt_idx]
                    confmat[gt_class - 1, num_classes] += 1


    # Convert the confusion matrix to a NumPy array
    confmat_np = confmat.cpu().numpy()
    confmat_no_bg = confmat_np[:-1, :-1]  # Removes the last row and column

    # Save the confusion matrix for future use
    os.makedirs(f'./outputs/confmat/{year}_{split}', exist_ok=True)
    np.save(confmat_file, confmat_no_bg)  # Save the matrix as a .npy file


#normalize the matrix row-wise
row_sums = confmat_no_bg.sum(axis=1, keepdims=True)
confmat_row_norm = confmat_no_bg / row_sums
# Plot the confusion matrix as a heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(confmat_row_norm, annot=True, fmt=".4f", cmap="Blues", cbar=False, annot_kws={"size": 12})

# Labels and title
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')

# Optionally, you can add labels for the background class too:
class_names = ['aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog', 'horse',
                            'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor']
# class_names = [f"Class {i}" for i in range(num_classes)] + ["Background"]
# Set tick positions and labels
plt.xticks(np.arange(n_classes), class_names, rotation=0, fontsize=12)
plt.yticks(np.arange(n_classes), class_names, rotation=0, fontsize=12)

plt.tight_layout()
# plt.show()
os.makedirs(f'./outputs/confmat/{year}_{split}', exist_ok=True)
plt.savefig(f'./outputs/confmat/{year}_{split}/confmat_{year}_{split}_normalized.png')
        