from datatypes import (
    TripleOutput,
    RPNOutput,
    FastOutput,
    Detection_cfg,
    Anchor_cfg,
    Rpn_cfg,
)

import utils as U

import torch
import torchvision
import torch.nn.init as init_weights
import torch.nn.functional as F
from torch.nn import (
    Conv2d, 
    Module, 
    Linear, 
    Sequential, 
    ReLU, 
    MaxPool2d,
    CrossEntropyLoss,
    SmoothL1Loss
)

def load_state_dict(model, base_model):
    m_dict = model.state_dict()
    b_dict = base_model.state_dict()
    b_dict = {k:v for k,v in b_dict.items() if k in m_dict}
    m_dict.update(b_dict)
    return model.load_state_dict(m_dict)

def freeze_module(module, freeze):
    for param in module.parameters():
        param.requires_grad = not freeze
    for child in module.children():  # Recursively freeze submodules
        freeze_module(child, freeze)

def roi_pool(x, rois, W=7, H=7, scale=1/16):
    n,c,h,w = x.shape
    roi_count = rois.shape[0]
    indices = rois[:, 0].int()

    # scale bboxes to appropriate size at the feature map
    bboxes = (rois[:, 1:5] * scale).round().int()
    pooled = torch.zeros((roi_count, c, H, W), dtype=x.dtype, device=x.device)
    for i in range(roi_count):
        # extract RoI from image
        batch_idx = indices[i]
        y1, x1, y2, x2 = bboxes[i].tolist()
        roi_ft = x[batch_idx:batch_idx+1, :, y1:y2+1, x1:x2+1]

        # pad RoI
        ft_h = roi_ft.size(2)
        ft_w = roi_ft.size(3)
        num_window_w = ((ft_w // W) + (1 if ft_w % W else 0))
        num_window_h = ((ft_h // H) + (1 if ft_h % H else 0))
        padding_w = num_window_w*W - ft_w
        padding_h = num_window_h*H - ft_h
        padded = F.pad(roi_ft, (0, padding_w, 0, padding_h), 'constant', 0)

        # compute max pool over HxW sub windows
        windows = F.unfold(padded, kernel_size=(H,W), stride=(H,W))
        # unfold: N, C, * -> N, C*k*k, number_of_windows
        windows = windows.view(1, c, H, W, -1)
        pooled[i] = windows.max(dim=4)[0]

    return pooled

class RoIPoolingLayer(Module):
    def __init__(self, H, W, scale):
        super().__init__()
        self.H = H
        self.W = W
        self.scale = scale

    def forward(self, x, rois):
        return roi_pool(x, rois, W=self.W, H=self.H, scale=self.scale)

class Backbone(Module):
    def __init__(self):
        super().__init__()
        base_model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.features = Sequential(*base_model.features[:-1])
        self.pooling_only = Sequential(*[m for m in self.features if isinstance(m, MaxPool2d)])
        self.classifier = Sequential(*base_model.classifier[:-1])

    def feature_map_size(self, x):
        x = self.pooling_only(x)
        return x.size()[-2:]

class TransformBBox(Module):
    def __init__(self):
        super().__init__()

    def forward(self, base_roi, transform):
        '''
        base_roi: (N, 4)
        transforms: (B, N, 4)
        returns: (B,N,4)
        '''
        base_roi = base_roi.unsqueeze(0)
        x1 = base_roi[:,:,0]
        y1 = base_roi[:,:,1]
        x2 = base_roi[:,:,2]
        y2 = base_roi[:,:,3]

        roi_w = x2-x1
        roi_h = y2-y1
        dx = transform[:,:, 0] * roi_w
        dy = transform[:,:, 1] * roi_h
        dw = transform[:,:, 2] * roi_w
        dh = transform[:,:, 3] * roi_h

        x1 = x1 + dx
        y1 = y1 + dy
        x2 = x2 + dx + dw
        y2 = y2 + dy + dh

        return torch.stack([x1,y1,x2,y2], dim=2)

class AnchorGenerator(Module):
    def __init__(self, cfg):
        super().__init__()
        self.anchor_dims = U.get_anchor_dims(cfg.areas, cfg.ratios)
        self.stride_len = cfg.stride_len
        self.lo_th = cfg.lo_th
        self.hi_th = cfg.hi_th
        self.n = cfg.n

    def forward(self, ft_h, ft_w):
        anchor_origins = U.get_anchor_origins(ft_w, ft_h, self.stride_len)
        anchor_dims = self.anchor_dims
        anchors = U.merge_anchor_origin_and_dim(anchor_origins, anchor_dims)
        return anchors

class StandaloneAnchorGenerator(AnchorGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)
        backbone = Backbone()
        self.pooling_only = backbone.pooling_only

    def forward(self, x):
        with torch.no_grad():
            x = self.pooling_only(x)
            outputs = super().forward(x.size(2), x.size(3))
            return outputs

class DetectionHead(Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.fc = backbone.classifier
        self.roi_pool = torchvision.ops.RoIPool((cfg.H, cfg.W), cfg.scale)
        # self.roi_pool = RoIPoolingLayer(cfg.H, cfg.W, cfg.scale)
        self.cls = Linear(cfg.hidden, cfg.n_classes)
        self.bbox = Linear(cfg.hidden, 4*cfg.n_classes)

        modules = [self.cls, self.bbox]
        init_stds = [cfg.init_cls_std, cfg.init_bbox_std]

        for m, std in zip(modules, init_stds):
            init_weights.normal_(m.weight, mean=0, std=std)
            init_weights.constant_(m.bias, cfg.init_bias)

    def forward(self, x, rois):
        x = self.roi_pool(x, rois)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        cls_logits = self.cls(x)
        cls_softmax = F.softmax(cls_logits, dim=1)
        bbox_reg = self.bbox(x)
        return TripleOutput(cls_logits, cls_softmax, bbox_reg)

class RPNHead(Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv = Conv2d(cfg.hidden, cfg.hidden, kernel_size=cfg.kernel_size, stride=1, padding=1)
        self.cls  = Conv2d(cfg.hidden, cfg.k*2, kernel_size=1, stride=1, padding=0)
        self.bbox = Conv2d(cfg.hidden, cfg.k*4, kernel_size=1, stride=1, padding=0)
        self.relu = ReLU()
        for m in [self.bbox, self.cls, self.conv]:
            init_weights.normal_(m.weight, mean=cfg.init_mean, std=cfg.init_std)

    def forward(self, x):
        if x.isnan().any():
            print("NaN found after rpn_input")

        B = x.size(0)
        x = self.conv(x)
        if x.isnan().any():
            print("NaN found after rpn_conv_1")
        x = self.relu(x)
        if x.isnan().any():
            print("NaN found after rpn_relu")

        cls_logits = self.cls(x)
        if cls_logits.isnan().any():
            print("NaN found after rpn_cls")

        cls_logits = cls_logits.permute(0,2,3,1).reshape(B, -1, 2)
        cls_logits = cls_logits - cls_logits.max(dim=2, keepdim=True)[0]# Prevent overflow


        #cls_logits = cls_logits.permute(0,2,3,1).contiguous().view(B, -1, 2)
        cls_softmax = F.softmax(cls_logits, dim=2)
        if cls_softmax.isnan().any():
            print("NaN found after rpn_softmax")

        bbox_reg = self.bbox(x)
        if bbox_reg.isnan().any():
            print("NaN found after rpn_bbox")
        bbox_reg = bbox_reg.permute(0,2,3,1).reshape(B, -1, 4)
        #bbox_reg = bbox_reg.permute(0,2,3,1).contiguous().view(B, -1, 4)

        return TripleOutput(cls_logits, cls_softmax, bbox_reg)

class RPN(Module):
    def __init__(self, rpn_cfg, anchor_cfg):
        super().__init__()
        backbone = Backbone()
        self.features = backbone.features
        self.rpn_layer = RPNHead(rpn_cfg)
        self.anchor_layer = AnchorGenerator(anchor_cfg)
        self.merge_layer = TransformBBox()

    def forward(self, x):
        x = self.features(x)
        outputs = self.rpn_layer(x)
        cls_logits = outputs.cls_logits
        cls_softmax = outputs.cls_softmax
        bbox_reg = outputs.bbox_reg
        anchors = self.anchor_layer(x.size(2), x.size(3))
        anchors = anchors.to(x.device)

        roi_proposals = self.merge_layer(anchors, outputs.bbox_reg)
        roi_proposals = roi_proposals.squeeze()

        return RPNOutput(cls_logits, cls_softmax, bbox_reg, anchors, roi_proposals)

class FastRCNN(Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = Backbone()
        self.features = backbone.features
        self.detection_layer = DetectionHead(backbone, cfg)
        self.merge_layer = TransformBBox()

    def forward(self, x, roi):
        x = self.features(x)
        outputs = self.detection_layer(x, rois)
        bbox_reg = outputs.bbox_reg
        n = rois.size(0)

        pred_cls = outputs.cls_softmax.argmax(dim=1)

        # treat (K+1) as batch size
        bbox_reg = bbox_reg.contiguous().view(n, -1, 4)
        bbox_reg = bbox_reg[torch.arange(n), pred_cls]
        roi_bbox = rois[:, 1:]

        locs = self.merge_layer(roi_bbox, bbox_reg.unsqueeze(0))
        locs = locs.squeeze()

        return FastOutput(outputs.cls_logits, outputs.cls_softmax, bbox_reg, locs, pred_cls)

class FasterRCNN(Module):
    def __init__(self, rpn_cfg, anchor_cfg, detection_cfg):
        super().__init__()
        backbone = Backbone()
        self.features = backbone.features
        self.rpn_layer = RPNHead(rpn_cfg)
        self.anchor_layer = AnchorGenerator(anchor_cfg)
        self.merge_layer = TransformBBox()
        self.detection_layer = DetectionHead(backbone, detection_cfg)

    def forward(self, x):
        im_h = x.size(2)
        im_w = x.size(3)
        x = self.features(x)
        rpn_out = self.rpn_layer(x)
        anchors = self.anchor_layer(x.size(2), x.size(3))
        anchors = anchors.to(x.device)

        roi_proposals = self.merge_layer(anchors, rpn_out.bbox_reg)
        roi_proposals = roi_proposals.squeeze()

        not_cross_idx = U.drop_cross_boundary_boxes(roi_proposals, im_w, im_h).to(x.device)
        filtered_proposals = roi_proposals[not_cross_idx]
        n_proposals = filtered_proposals.size(0)
        indices = torch.full((n_proposals,), 0, device=x.device)
        rois = torch.cat([
            indices.unsqueeze(1),
            filtered_proposals,
        ], dim=1)

        det_out = self.detection_layer(x, rois)
        bbox_reg = det_out.bbox_reg
        pred_cls = det_out.cls_softmax.argmax(dim=1)
        bbox_reg = bbox_reg.contiguous().view(n_proposals, -1, 4)
        bbox_reg = bbox_reg[torch.arange(n_proposals), pred_cls]

        locs = self.merge_layer(filtered_proposals, bbox_reg.unsqueeze(0))
        locs = locs.squeeze()

        return {
            'rpn': rpn_out,
            'detection': det_out,
            'anchors': anchors,
            'roi_proposals': roi_proposals,
            'filtered_proposals': filtered_proposals,
            'locs': locs,
            'pred_cls': pred_cls,
            'bbox_reg': bbox_reg,
        }

class RPNLoss(Module):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam
        self.cls_loss = CrossEntropyLoss()
        self.bbox_loss = SmoothL1Loss()

    def forward(self, cls_pred, cls_target, bbox_pred, bbox_target):
        cls_loss = self.cls_loss(cls_pred, cls_target)
        bbox_target = bbox_target[cls_target == 1]
        bbox_pred = bbox_pred[cls_target == 1]
        if cls_target.sum() == 0:
            bbox_loss = torch.tensor(0.0, dtype=cls_loss.dtype, device=cls_loss.device)
        else:
            bbox_loss = self.bbox_loss(bbox_pred, bbox_target)

        loss = cls_loss + self.lam * bbox_loss
        return loss

class RPNLoss_v2(Module):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam
        self.cls_loss = CrossEntropyLoss()
        self.bbox_loss = SmoothL1Loss()

    '''
    Fixed incorrect indexing
    '''
    def forward(self, outputs, y):
        gt_bboxes = y['bboxes']
        im_w = y['width']
        im_h = y['height']

        cls_logits = outputs.cls_logits
        anchors = outputs.anchors
        roi_proposals = outputs.roi_proposals

        # subset
        not_cross_idx = U.drop_cross_boundary_boxes(anchors, im_w, im_h).to(anchors.device)
        anchors = anchors[not_cross_idx]
        cls_logits = cls_logits[:, not_cross_idx]
        roi_proposals = roi_proposals[not_cross_idx]

        # subset of subset
        sampled_idx, gt_idx, objectness = U.sample_anchors(
            anchors, gt_bboxes, .3, .7, 256
        )

        cls_logits = cls_logits[:, sampled_idx]

        # cls loss
        cls_loss = self.cls_loss(cls_logits[0], objectness)
        # bbox loss
        if objectness.sum() > 0:
            pos_idx = sampled_idx[objectness > 0]
            pos_gt_idx = gt_idx[objectness > 0]
            pos_proposals = roi_proposals[pos_idx]
            pos_anchors = anchors[pos_idx]
            pos_gt_box = gt_bboxes[pos_gt_idx]
            # parameterized as per paper
            t_pred = U.parameterize_bbox(pos_proposals, pos_anchors)
            t_gt = U.parameterize_bbox(pos_gt_box, pos_anchors)
            bbox_loss = self.bbox_loss(t_pred, t_gt)
        else:
            bbox_loss = 0.0

        loss = cls_loss + self.lam * bbox_loss

        return loss


class MultiTaskLoss(Module):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam
        self.cls_loss = CrossEntropyLoss()
        self.bbox_loss = SmoothL1Loss()
    
    def forward(self, cls_pred, cls_target, bbox_pred, bbox_target):
        cls_loss = self.cls_loss(cls_pred, cls_target)
        if bbox_pred.size(0) == 0:
            bbox_loss = torch.tensor(0.0, dtype=cls_loss.dtype, device=cls_loss.device)
        else:
            bbox_loss = self.bbox_loss(bbox_pred, bbox_target)

        loss = cls_loss + self.lam * bbox_loss
        return loss
