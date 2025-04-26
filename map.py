import os
import json

import torch
from torchvision import transforms
from torchvision.datasets import VOCDetection

import matplotlib.pyplot as plt

import argparse

from models import FasterRCNN

CLASS2IDX = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 
        'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9,
        'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13,
        'motorbike': 14, 'person': 15, 'pottedplant': 16,
        'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}

PRED_BOX_NUM = 300

IOU_THRESHOLD = 0.5

def construct_gt_box_list_by_class(year='2007', image_set='test', class2idx=CLASS2IDX):
    '''
    Returns all gt boxes organized by class:
    [[{image_id: int, gt_coordinates: list of 4 coordinates}, ...], [], [], ...]
    
    A list of 20 lists of gt boxes.
    '''
    save_path = f'./data/voc_{year}_{image_set}.json'

    # If file exists, load and return
    if os.path.exists(save_path):
        print("Ground truth file exists, loading directly...")
        with open(save_path, 'r') as f:
            gt_box_list_by_class = json.load(f)
        return gt_box_list_by_class

    # Otherwise, construct the gt box list
    print("Constructing ground truth list...")
    gt_box_list_by_class = [[] for _ in range(20)]
    voc_test = VOCDetection(root='./data', year=year, image_set=image_set, download=False)
    for image_id, image in enumerate(voc_test):
        print(f"Processing image {image_id}...")
        obj_list = image[1]['annotation']['object']
        if isinstance(obj_list, dict):  # If only one object, wrap it into a list
            obj_list = [obj_list]
        for obj in obj_list:
            xmin = float(obj['bndbox']['xmin'])
            ymin = float(obj['bndbox']['ymin'])
            xmax = float(obj['bndbox']['xmax'])
            ymax = float(obj['bndbox']['ymax'])
            gt_box = {'image_id': image_id, 'gt_coordinates': [xmin, ymin, xmax, ymax]}
            gt_box_list_by_class[class2idx[obj['name']] - 1].append(gt_box)

    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(gt_box_list_by_class, f)

    print("Ground truth box list constructed and saved.")

    return gt_box_list_by_class

'''
List of prediction boxes:
[[{image_id: int, pred_coordinates: list of 4 coordinates, confidence_score: int}, ...], [], [], ...]

A list of 20 lists of predicted boxes.

Below are the filtering functions for the predicted boxes.

'''

def filter_single_inference_per_image(image_id, outputs, current_list_of_pred, pred_box_num=PRED_BOX_NUM):
    '''
    300 boxes per image
    '''

    cls_pred = outputs[0]['det_cls_pred'].cpu().tolist()
    bbox_pred = outputs[0]['det_bbox_pred'].cpu().tolist()
    score = outputs[0]['det_score'].cpu().tolist()

    top_indices = sorted(range(len(score)), key=lambda i: score[i], reverse=True)[:pred_box_num]
    filter_pred_bbox = [bbox_pred[i] for i in top_indices]
    filter_score = [score[i] for i in top_indices]
    for box_idx, pred_bbox in enumerate(filter_pred_bbox):
        pred_box = {'image_id': image_id, 'pred_coordinates': pred_bbox, 'confidence_score': filter_score[box_idx]}
        pred_class = cls_pred[top_indices[box_idx]]
        current_list_of_pred[pred_class - 1].append(pred_box)

    return current_list_of_pred



def filter_single_inference_per_class(image_id, outputs, current_list_of_pred, pred_box_num=PRED_BOX_NUM):
    '''
    300 boxes per class per image
    '''

    cls_pred = outputs[0]['det_cls_pred'].cpu().tolist()
    bbox_pred = outputs[0]['det_bbox_pred'].cpu().tolist()
    score = outputs[0]['det_score'].cpu().tolist()

    for class_idx in range(1, 21):
        # pred_bbox_single_class = bbox_pred[cls_pred == class_idx]
        # pred_score_single_class = score[cls_pred == class_idx]
        pred_bbox_single_class = [b for b, c in zip(bbox_pred, cls_pred) if c == class_idx]
        pred_score_single_class = [s for s, c in zip(score, cls_pred) if c == class_idx]

        top_indices = sorted(range(len(pred_score_single_class)), key=lambda i: pred_score_single_class[i], reverse=True)[:pred_box_num]
        filter_pred_bbox_single_class = [pred_bbox_single_class[i] for i in top_indices]
        filter_score_single_class = [pred_score_single_class[i] for i in top_indices]
        for box_idx, pred_bbox in enumerate(filter_pred_bbox_single_class):
            pred_box = {'image_id': image_id, 'pred_coordinates': pred_bbox, 'confidence_score': filter_score_single_class[box_idx]}
            current_list_of_pred[class_idx - 1].append(pred_box)

    return current_list_of_pred

def filter_single_inference_confidence(image_id, outputs, current_list_of_pred, confidence_threshold=0.1):
    '''
    predictions that have confidence score above threshold
    '''
    cls_pred = outputs[0]['det_cls_pred'].cpu().tolist()
    bbox_pred = outputs[0]['det_bbox_pred'].cpu().tolist()
    score = outputs[0]['det_score'].cpu().tolist()

    filter_pred_bbox = [b for b, s in zip(bbox_pred, score) if s > confidence_threshold]
    filter_score = [s for s in score if s > confidence_threshold]
    filter_cls_pred = [c for c, s in zip(cls_pred, score) if s > confidence_threshold]
    for box_idx, pred_bbox in enumerate(filter_pred_bbox):
        pred_box = {'image_id': image_id, 'pred_coordinates': pred_bbox, 'confidence_score': filter_score[box_idx]}
        pred_class = filter_cls_pred[box_idx]
        current_list_of_pred[pred_class - 1].append(pred_box)

    return current_list_of_pred


def filter_single_inference_entire_set(image_id, outputs, current_list_of_pred, filter_threshold, pred_box_num=PRED_BOX_NUM):
    '''
    300 boxes per class for the entire set
    '''
    cls_pred = outputs[0]['det_cls_pred'].cpu().tolist()
    bbox_pred = outputs[0]['det_bbox_pred'].cpu().tolist()
    score = outputs[0]['det_score'].cpu().tolist()

    for class_idx in range(1, 21):
        pred_bbox_single_class = [b for b, c in zip(bbox_pred, cls_pred) if c == class_idx]
        pred_score_single_class = [s for s, c in zip(score, cls_pred) if c == class_idx]
        filter_pred_bbox_single_class = [
            b for b, s in zip(pred_bbox_single_class, pred_score_single_class)
            if s > filter_threshold[class_idx - 1]
        ]
        filter_score_single_class = [
            s for s in pred_score_single_class
            if s > filter_threshold[class_idx - 1]
        ]

        for box_idx, pred_bbox in enumerate(filter_pred_bbox_single_class):
            pred_box = {'image_id': image_id, 'pred_coordinates': pred_bbox, 'confidence_score': filter_score_single_class[box_idx]}
            current_list_of_pred[class_idx - 1].append(pred_box)

        # truncate prediction list and update thresold
        sorted_list_of_pred_bbox = sorted(current_list_of_pred[class_idx - 1], key=lambda x: x['confidence_score'], reverse=True)
        if len(sorted_list_of_pred_bbox) > pred_box_num:
            current_list_of_pred[class_idx - 1] = sorted_list_of_pred_bbox[:pred_box_num]
            filter_threshold[class_idx - 1] = sorted_list_of_pred_bbox[pred_box_num - 1]['confidence_score']

    return current_list_of_pred, filter_threshold

def construct_pred_box_list_by_class(model, device, year='2007', image_set='test'):
    # Initialize
    print("Constructing prediction list...")
    filter_threshold = [0 for _ in range(20)]
    list_of_pred = [[] for _ in range(20)]
    voc_test = VOCDetection(root='./data', year=year, image_set=image_set, download=False)
    for image_id, image in enumerate(voc_test):
        print(f"Processing image {image_id}...")
        img = image[0]
        to_tensor = transforms.ToTensor()
        tensor_img = to_tensor(img)
        tensor_img = tensor_img.unsqueeze(0)
        tensor_img = tensor_img.to(device)

        model.eval()

        with torch.no_grad():
            outputs = model((tensor_img, None))

        list_of_pred, filter_threshold = filter_single_inference_entire_set(image_id, outputs, list_of_pred, filter_threshold) # can be modified

    print("Prediction list constructed.")
    return list_of_pred

def compute_iou(boxes1, boxes2):
    """
    Compute pairwise IoU between two lists of boxes.
    
    Args:
        boxes1: List of boxes, each box is [xmin, ymin, xmax, ymax]
        boxes2: List of boxes, same format

    Returns:
        A 2D list where result[i][j] is the IoU between boxes1[i] and boxes2[j]
    """
    result = []
    epsilon = 1e-7

    for box1 in boxes1:
        row = []
        for box2 in boxes2:
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])

            if x_right <= x_left or y_bottom <= y_top:
                row.append(0.0)
                continue

            intersection = (x_right - x_left) * (y_bottom - y_top)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection

            iou = intersection / (union + epsilon)
            row.append(iou)
        result.append(row)
    
    return result

def compute_precision_recall_ap(class_idx, gt_boxes, pred_boxes, iou_threshold=IOU_THRESHOLD, save_pr_plot=True):
    # sort predictions by confidence
    pred_boxes = sorted(pred_boxes, key=lambda x: x['confidence_score'], reverse=True)
    
    image_to_gt = {}
    for gt in gt_boxes:
        image_id = gt['image_id']
        if image_id not in image_to_gt:
            image_to_gt[image_id] = []
        image_to_gt[image_id].append({'coords': gt['gt_coordinates'], 'used': False})
    
    tp = []
    fp = []
    total_gt = len(gt_boxes)

    for pred in pred_boxes:
        image_id = pred['image_id']
        pred_coord = pred['pred_coordinates']
        
        gt_in_image = image_to_gt.get(image_id, [])
        gt_coords = [gt['coords'] for gt in gt_in_image]

        if not gt_coords:
            # No GTs for this image
            fp.append(1)
            tp.append(0)
            continue
        
        ious = compute_iou([pred_coord], gt_coords)[0]  # 1 x N
        best_iou_idx = max(range(len(ious)), key=lambda i: ious[i])
        best_iou = ious[best_iou_idx]

        if best_iou >= iou_threshold and not gt_in_image[best_iou_idx]['used']:
            tp.append(1)
            fp.append(0)
            gt_in_image[best_iou_idx]['used'] = True  # Mark GT as used
        else:
            fp.append(1)
            tp.append(0)
    
    # compute cumulative TP and FP
    tp_cumsum = []
    fp_cumsum = []
    tp_sum = 0
    fp_sum = 0

    for t, f in zip(tp, fp):
        tp_sum += t
        fp_sum += f
        tp_cumsum.append(tp_sum)
        fp_cumsum.append(fp_sum)

    precisions = [tp_cumsum[i] / (tp_cumsum[i] + fp_cumsum[i]) for i in range(len(tp_cumsum))]
    recalls = [tp_cumsum[i] / total_gt for i in range(len(tp_cumsum))]

    # compute AP
    # 11-point interpolation
    ap = 0.0
    for recall_threshold in [i/10 for i in range(11)]:
        precisions_at_recall = [p for p, r in zip(precisions, recalls) if r >= recall_threshold]
        if precisions_at_recall:
            ap += max(precisions_at_recall)
    ap /= 11

    # plot Precision-Recall curve
    # plt.show()
    if save_pr_plot:
        
        os.makedirs('./outputs/pr_curves', exist_ok=True)

        plt.figure(figsize=(8,6))
        plt.plot(recalls, precisions, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.grid(True)
        plt.savefig(f'./outputs/pr_curves/pr_curve_class_{class_idx}.png')

    return ap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate mAP and plot Precision-Recall curve.")

    parser.add_argument('--year', type=str, default='2007', help='Year of the VOC dataset')
    parser.add_argument('--image_set', type=str, default='test', help='Image set (train, trainval, val, test)')
    parser.add_argument('--model_path', type=str, default='./outputs/checkpoint_step4_80000.pt', help='Path to the model checkpoint')

    args = parser.parse_args()
    
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gt_box_list_by_class = construct_gt_box_list_by_class(year=args.year, image_set=args.image_set)

    # Load model
    model = FasterRCNN()
    model.to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state['model'])

    pred_box_list_by_class = construct_pred_box_list_by_class(model, device, year=args.year, image_set=args.image_set)

    ap_per_class = []
    for idx in range(20):
        ap = compute_precision_recall_ap(idx + 1, gt_box_list_by_class[idx], pred_box_list_by_class[idx], save_pr_plot=True)

        print(f'Class {idx + 1} AP: {ap}')
        ap_per_class.append(ap)

    print(f'mAP: {sum(ap_per_class) / len(ap_per_class)}')