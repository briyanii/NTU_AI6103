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
    iou = intersection / union.clamp(min=1e-6)  # Avoid division by zero
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
    N = boxes1.size(0)
    M = boxes2.size(0)

    # Extract box coordinates
    x1_1, y1_1, x2_1, y2_1 = boxes1.unsqueeze(1).unbind(dim=2) # Shapes (N, 1)
    x1_2, y1_2, x2_2, y2_2 = boxes2.unsqueeze(0).unbind(dim=2) # Shapes (1, M)

    # Compute intersection (top-left and bottom-right corners)
    inter_x1 = torch.max(x1_1, x1_2)
    inter_y1 = torch.max(y1_1, y1_2)
    inter_x2 = torch.min(x2_1, x2_2)
    inter_y2 = torch.min(y2_1, y2_2)

    # Compute intersection area
    inter_w = (inter_x2 - inter_x1).clamp(min=0)  # Width cannot be negative
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    intersection = inter_w * inter_h  # Shape: (N, M)

    # Compute area of both sets of boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)  # Shape: (N,1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)  # Shape: (1,M)

    # Compute union area
    union = area1 + area2 - intersection  # Shape: (N, M)

    # Compute IoU
    iou = intersection / union.clamp(min=1e-6)  # Avoid division by zero
    return iou


def rand_sample(arr, tgt_sz):
    n = arr.size(0)
    idx = torch.randperm(n)[:tgt_sz]
    return idx, arr[idx]

def sample_rois(rois, gt_cls, gt_box, pos_th, neg_lo, neg_hi, sample_size, pos_ratio):
    '''
        prereq: rois and gt on same device
        find pos and neg rois based on iou
        for pos rois
            sample up to PR% of the sample_size
        for neg rois
            sample up to remaining quota
    '''
    # prereq: both rois and gt on same device
    n = rois.size(0)
    ious = compute_iou(rois, gt_box)

    # assign pos / neg labels
    max_iou_per_roi, gt_index_per_roi = ious.max(dim=1)
    roi_index_per_gt = ious.argmax(dim=0)

    is_pos = max_iou_per_roi >= pos_th
    is_neg = (max_iou_per_roi >= neg_lo) & (max_iou_per_roi < neg_hi)
    n_pos = is_pos.sum().item()
    n_neg = is_neg.sum().item()

    pos_sample_size = min(n_pos, int(sample_size*pos_ratio))
    neg_sample_size = min(n_neg, sample_size - pos_sample_size)

    indices = torch.arange(0, n)
    pos_idx = rand_sample(indices[is_pos], pos_sample_size)[1]
    neg_idx = rand_sample(indices[is_neg], neg_sample_size)[1]
    sample_idx = torch.cat([pos_idx, neg_idx])

    gt_cls = gt_cls[gt_index_per_roi]
    gt_cls[neg_idx] = 0
    gt_cls = gt_cls[sample_idx]

    gt_box = gt_box[gt_index_per_roi]
    gt_box = gt_box[sample_idx]

    rois = rois[sample_idx]

    return rois, gt_cls, gt_box

def cat_val(arr, val):
    n = arr.size(0)
    vals = torch.full((n,1), val)
    return torch.cat([vals, arr], dim=1)

def sample_anchors(anchors, gt, lo_th, hi_th, sample_size):
    # prereq: both anchors and gt on same device
    ious = compute_iou(anchors, gt)

    # assign pos / neg labels
    max_iou_per_anchor, gt_index_per_anchor = ious.max(dim=1)
    anchor_index_per_gt = ious.argmax(dim=0)

    is_pos = max_iou_per_anchor > hi_th
    is_neg = max_iou_per_anchor < lo_th
    is_pos[anchor_index_per_gt] = True
    is_neg[anchor_index_per_gt] = False

    n_pos = is_pos.sum()
    n_neg = is_neg.sum()

    pos_sample_size = min(n_pos.item(), sample_size//2)
    neg_sample_size = sample_size - pos_sample_size

    indices = torch.arange(0, anchors.size(0))
    pos_idx = rand_sample(indices[is_pos], pos_sample_size)
    neg_idx = rand_sample(indices[is_neg], neg_sample_size)
    sample_idx = torch.cat([pos_idx, neg_idx])

    gt_idx = gt_index_per_anchor[sample_idx]
    objectness = torch.zeros_like(sample_idx)
    objectness[:n_pos] = 1

    return sample_idx, gt_idx, objectness

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
    bboxes = bboxes.cpu()

    lower_bnd = torch.zeros((4,))
    upper_bnd = torch.Tensor([im_w, im_h, im_w, im_h])
    condition = (bboxes >= lower_bnd) & (bboxes <= upper_bnd)

    idx = torch.where(condition)[0]

    return idx
