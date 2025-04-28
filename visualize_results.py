# Visualize the best matched prediction for every gt
import argparse
import os
import torch
import matplotlib.pyplot as plt
import utils as U
from dataset import VOCDataset
from dataloader import get_dataloader
from models import FasterRCNN

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--n_classes', type=int, default=20)
parser.add_argument('--iou_threshold', type=float, default=.5)
parser.add_argument('--n_proposals', type=int, default=300)
parser.add_argument('--year', type=int, default=2007)
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--model_weight', type=str, default='./outputs/checkpoint_step4_80000.pt')

args = parser.parse_args()
n_classes = args.n_classes
n_proposals = args.n_proposals
iou_th = args.iou_threshold
year = args.year
split = args.split
model_weight = args.model_weight

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight_path = model_weight
state = torch.load(weight_path, map_location='cpu')
model = FasterRCNN()
model.load_state_dict(state['model'])
model = model.to(device)
model.eval()

dataset = VOCDataset(year, split, load=False)
# original_dataset = VOCDataset(year, split, load=True, normalize=False, scale=False, augment=False)
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

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog', 'horse',
                            'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor']


mean = torch.tensor([0.485, 0.456, 0.406])
std  = torch.tensor([0.229, 0.224, 0.225])
def denormalize(img_norm):
    """
    img_norm: Tensor[C,H,W] in normalized space
    returns:  Tensor[C,H,W] in [0,1]
    """
    img = img_norm.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1)

with torch.no_grad():
    for step, batch in enumerate(dataloader):
        gt_boxes = []
        best_pred_boxes = []
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

        # # filtering 300 bboxes per image
        # indices = score.argsort(descending=True)[:n_proposals]
        # filter_cls_pred = cls_pred[indices]
        # filter_bbox_pred = bbox_pred[indices]
        # filter_score = score[indices]


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

            # record every gt box
            for gt_bbox in _bbox_gt:
                gt_boxes.append({'class': cls_id, 'bbox': gt_bbox})

            # compute ious
            ious = U.compute_iou(_bbox_pred, _bbox_gt)
            N,M = ious.shape

            if N == 0 or  M == 0:
                continue

            max_iou_per_gt, pred_idx_per_gt = ious.max(dim=0)

            # record best pred box
            best_pred_bboxes_single_class = _bbox_pred[pred_idx_per_gt]
            for best_pred_bbox in best_pred_bboxes_single_class:
                best_pred_boxes.append({'class': cls_id, 'bbox': best_pred_bbox})

        # visualize and save
        # tensor_img = original_dataset[step]['image']
        tensor_img = imgs[0]
        tensor_img = denormalize(tensor_img)

        to_pil = transforms.ToPILImage()
        pil_img = to_pil(tensor_img)

        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.load_default()

        # draw gt boxes
        for gt_box in gt_boxes:
            gt_class_name = object_categories[gt_box['class'] - 1]
            gt_coor = gt_box['bbox'].tolist()
            draw.rectangle(gt_coor, outline="green", width=4)
            draw.text((gt_coor[0], gt_coor[1] - 10), gt_class_name, fill="green", font=font)

        # draw pred boxes
        for pred_box in best_pred_boxes:
            pred_class_name = object_categories[pred_box['class'] - 1]
            box_coor = pred_box['bbox'].tolist()
            # print(box_coor) # debug
            draw.rectangle(box_coor, outline="red", width=4)
            draw.text((box_coor[0], box_coor[1] - 10), pred_class_name, fill="red", font=font)

        # save the image
        os.makedirs(f'./outputs/results/{year}_{split}', exist_ok=True)
        savepath = f'./outputs/results/{year}_{split}/{dataset[step]['filename']}'
        pil_img.save(savepath)
        print(f"save detection result to {savepath}")