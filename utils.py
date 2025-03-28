import torch
import  numpy as np

def iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    
    box1, box2: (x1, y1, x2, y2) in tensor format.
    Returns IoU score.
    """
    # Get intersection box coordinates
    ix1 = torch.max(box1[0], box2[0])
    iy1 = torch.max(box1[1], box2[1])
    ix2 = torch.min(box1[2], box2[2])
    iy2 = torch.min(box1[3], box2[3])
    
    # Compute intersection area
    w = torch.clamp(ix2 - ix1, min=0)
    h = torch.clamp(iy2 - iy1, min=0)
    intersection = w * h
    
    # Compute area of both boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute union area
    union = area1 + area2 - intersection
    
    # Compute IoU
    iou = intersection / union
    return iou
    

def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two batches of bounding boxes.
    
    Args:
        boxes1 (Tensor): Shape (N, 4) with (x1, y1, x2, y2)
        boxes2 (Tensor): Shape (M, 4) with (x1, y1, x2, y2)
    Returns:
        Tensor: IoU matrix of shape (N, M), where IoU[i, j] is IoU of boxes1[i] with boxes2[j]
    """
    # Extract box coordinates
    x1_1, y1_1, x2_1, y2_1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]  # Shape: (N,)
    x1_2, y1_2, x2_2, y2_2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]  # Shape: (M,)
    
    # Compute intersection (top-left and bottom-right corners)
    inter_x1 = torch.max(x1_1[:, None], x1_2[None, :])  # Shape: (N, M)
    inter_y1 = torch.max(y1_1[:, None], y1_2[None, :])
    inter_x2 = torch.min(x2_1[:, None], x2_2[None, :])
    inter_y2 = torch.min(y2_1[:, None], y2_2[None, :])
    
    # Compute intersection area
    inter_w = (inter_x2 - inter_x1).clamp(min=0)  # Width cannot be negative
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    intersection = inter_w * inter_h  # Shape: (N, M)
    
    # Compute area of both sets of boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)  # Shape: (N,)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)  # Shape: (M,)
    
    # Compute union area
    union = area1[:, None] + area2[None, :] - intersection  # Shape: (N, M)
    
    # Compute IoU
    iou = intersection / union.clamp(min=1e-6)  # Avoid division by zero
    
    return iou

def sample_rois(proposals, gt_bbox, pos_th, neg_th_lo, neg_th_hi, sample_size, pos_ratio):

    pos_sample_size = int(sample_size * pos_ratio)

    ious = compute_iou(proposals, gt_bbox)
    case_1 = (ious >= pos_th)
    case_2 = (ious >= neg_th_lo) & (ious < neg_th_hi)
    case_2 = case_2.all(dim=1)
    case_1 = case_1.any(dim=1)

    labels = torch.where(case_1, 1, 0)

    pos_size = case_1.sum()
    neg_size = case_2.sum()

    pos_idx = torch.randperm(pos_size)[:pos_sample_size]
    pos_idx = torch.where(case_1)[0][pos_idx]
    pos_sample_size = pos_idx.size(0)

    neg_idx = torch.randperm(neg_size)[:sample_size - pos_sample_size]
    neg_idx = torch.where(case_2)[0][neg_idx]
    neg_sample_size = neg_idx.size(0)

    pos_samples = proposals[pos_idx]
    neg_samples = proposals[neg_idx]
    samples = torch.cat([pos_samples, neg_samples], dim=0)
    sample_indices = torch.cat([neg_idx, pos_idx], dim=0)
    sample_gt_idx = ious[sample_indices].argmax(dim=1)
    sample_labels = labels[sample_indices]
    sample_gt_idx = torch.where(sample_labels==1, sample_gt_idx, -1)

    return samples, sample_indices, sample_labels, sample_gt_idx

    
    
def sample_anchors(anchors, gt, lo_th, hi_th, sample_size):
    pos_sample_size = sample_size//2
    ious = compute_iou(anchors, gt)

    # assign pos / neg labels
    case_0 = ious.argmax(dim=0).to(anchors.device)
    case_1 = (ious > hi_th).any(dim=1).to(anchors.device)
    case_2 = (ious < lo_th).all(dim=1).to(anchors.device)
    labels = torch.where(case_2, -1, 0).to(anchors.device)
    labels = torch.where(case_1, 1, labels)
    labels[case_0] = 1
    
    is_pos = (labels == 1)
    is_neg = (labels == -1)
    n_pos = is_pos.sum()
    n_neg = is_neg.sum()
    
    # sample up to n//2
    pos_idx = torch.randperm(n_pos, device=anchors.device)
    pos_idx = pos_idx[:pos_sample_size]
    pos_idx = torch.where(labels==+1)[0][pos_idx]
    pos_sample_size = pos_idx.size(0)
    # sample remaining to make up to n
    neg_idx = torch.randperm(n_neg, device=anchors.device)
    neg_idx = neg_idx[:sample_size - pos_sample_size]
    neg_idx = torch.where(labels==-1)[0][neg_idx]
    
    neg_anchors = anchors[neg_idx]
    pos_anchors = anchors[pos_idx]
    
    sampled_anchors = torch.cat([pos_anchors, neg_anchors], dim=0)
    sampled_indices = torch.cat([pos_idx, neg_idx], dim=0)
    sampled_labels = labels[sampled_indices]
    sampled_labels = torch.where(sampled_labels==1, 1, 0).to(anchors.device)
    sampled_gt_idx = ious[sampled_indices].argmax(dim=1)
    
    return sampled_anchors, sampled_indices, sampled_labels, sampled_gt_idx

def get_anchor_dims(areas, ratios):
    '''
    w,h for anchors
    
    ratio = h_r, w_r
    A = area
    w x h = A
    h/w = h_r / w_r
    h = w * h_r / w_r
    w * w * h_r/w_r = A
    w**2 = A * w_r / h_r
    w = sqrt( A * w_r / h_r)
    h = h_r / w_r * w
    '''
    anchor_dims = []
    n_areas = len(areas)
    n_ratios = len(ratios)
    anchor_dims = torch.zeros((n_areas * n_ratios, 2), dtype=torch.float32)
    for i in range(n_areas):
        for j in range(n_ratios):
            idx = i * n_areas + j
            h_r, w_r = ratios[j]
            area = areas[i]
            w = np.sqrt(area * w_r / h_r)
            h = w * h_r / w_r
            anchor_dims[idx, 0] = w
            anchor_dims[idx, 1] = h
    return anchor_dims

def get_anchor_origins(ft_w, ft_h, stride_len):
    '''
    origin (cx,cy) for anchors
    '''
    origin_x = torch.arange(0, ft_w*stride_len, stride_len)
    origin_y = torch.arange(0, ft_h*stride_len, stride_len)
    origin_x, origin_y = torch.meshgrid(origin_x, origin_y, indexing='ij')
    origin_x = origin_x.ravel()
    origin_y = origin_y.ravel()
    anchor_origins = torch.vstack([origin_x, origin_y]) + stride_len / 2
    return anchor_origins.T

def merge_anchor_origin_and_dim(anchor_origins, anchor_dims):
    '''
    sum each pair to  get
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    sum using broadcasting
    '''
    anchor_origins = anchor_origins.unsqueeze(1)
    anchor_dims = anchor_dims.unsqueeze(0)
    anchors = torch.cat([
        anchor_origins - anchor_dims/2,
        anchor_origins + anchor_dims/2,
    ], dim=2)
    anchors = anchors.reshape(-1,4)
    return anchors

def xyxy_2_xywh(xyxy):
    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]
    w = x2 - x1
    h = y2 - y1
    
    xywh = torch.stack([x1,y1,w,h], dim=1)
    return xywh


def xywh_2_xyxy(xywh):
    x1 = xywh[:, 0]
    y1 = xywh[:, 1]
    w = xywh[:, 2]
    h = xywh[:, 3]
    
    x2 = x1 + w
    y2 = y1 + h
    
    xyxy = torch.stack([x1,y1,x2,y2], dim=1)
    return xyxy

def parameterize_bbox(bbox, anchor):
    if bbox.size(0) == 0:
        return torch.empty((0, 4))

    anchor = xyxy_2_xywh(anchor)
    bbox = xyxy_2_xywh(bbox)
    anchor_w = anchor[:, 2]
    anchor_h = anchor[:, 3]
    tx = (bbox[:, 0] - anchor[:, 0]) / anchor_w
    ty = (bbox[:, 1] - anchor[:, 1]) / anchor_h
    tw = torch.log1p(bbox[:, 2] / anchor_w - 1)
    th = torch.log1p(bbox[:, 3] / anchor_h - 1)

    txywh = torch.stack([tx,ty,tw,th], dim=1)
    return txywh


def clip_bboxes(bboxes, im_w, im_h):
    bboxes[:, 0] = torch.clamp(bboxes[:, 0], 0, im_w)
    bboxes[:, 1] = torch.clamp(bboxes[:, 1], 0, im_h)
    bboxes[:, 2] = torch.clamp(bboxes[:, 2], 0, im_w)
    bboxes[:, 3] = torch.clamp(bboxes[:, 3], 0, im_h)
    
    return bboxes


def drop_cross_boundary_boxes(bboxes, im_w, im_h):
    '''
    drop cross-boundary bounding boxes
    '''
    condition = (bboxes[:,0]>=0) & (bboxes[:,1]>=0) & (bboxes[:,2]<=im_w) & (bboxes[:,3]<=im_h)
    bboxes = bboxes[condition]
    return bboxes
