import utils as U
from config import config
import torch
import torchvision
import torch.nn.init as init_weights
from torchvision.ops import nms
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
    def forward(self, bbox, transform):
        '''
        R-CNN; https://arxiv.org/pdf/1311.2524
        Appendix C

        bbox: (B,N,4)
        transforms: (B,N,4)
        returns: (B,N,4)
        '''
        with torch.no_grad():
            x1,y1,x2,y2 = bbox.unbind(dim=2)
            w = x2-x1
            h = y2-y1
            cx = (x1 + x2)/2
            cy = (y1 + y2)/2

            # output of bbox regression layer
            # dx,dy -> scale invariant translation
            # dw,dh -> log space translation
            dx,dy,dw,dh = transform.unbind(dim=2)
            _w = w * torch.exp(dw)
            _h = h * torch.exp(dh)
            _cx = w * dx + cx
            _cy = h * dy + cy

            _x1 = _cx - _w/2
            _y1 = _cy - _h/2
            _x2 = _cx + _w/2
            _y2 = _cy + _h/2

            return torch.stack([_x1,_y1,_x2,_y2], dim=2)

class AnchorGenerator(Module):
    def __init__(self, 
        config=config,
    ):
        super().__init__()
        self.config = config
        self.anchor_dims = U.get_anchor_dims(config['areas'], config['ratios'])

    def forward(self, ft_h, ft_w):
        anchor_origins = U.get_anchor_origins(ft_w, ft_h, self.config['stride_len'])
        anchor_dims = self.anchor_dims
        anchors = U.merge_anchor_origin_and_dim(anchor_origins, anchor_dims)
        return anchors

class DetectionHead(Module):
    def __init__(self,
        fc,
        config=config,
    ):
        super().__init__()
        self.config = config
        self.roi_pool = torchvision.ops.RoIPool((config['pool_H'],config['pool_W']), 1/config['stride_len'])
        self.fc = fc
        self.cls = Linear(config['det_hidden'], config['det_classes'])
        self.bbox = Linear(config['det_hidden'], config['det_classes']*4)

        modules = [self.cls, self.bbox]
        init_stds = [config['det_std_cls'], config['det_std_box']]

        for m, std in zip(modules, init_stds):
            init_weights.normal_(m.weight, mean=0, std=std)
            init_weights.constant_(m.bias, config['det_bias'])

    def forward(self, x):
        '''
            x: (B, ...)
        '''
        x = self.fc(x)
        cls_logits = self.cls(x) # (K,21)
        cls_softmax = F.softmax(cls_logits, dim=1)

        bbox_reg = self.bbox(x) # (K,4*21)
        bbox_reg = bbox_reg.contiguous().view(x.size(0), -1, 4) # (K, 21, 4)
        return cls_logits, cls_softmax, bbox_reg

class RPNHead(Module):
    def __init__(self,
        config=config,
    ):
        super().__init__()
        self.config = config
        self.conv = Conv2d(config['rpn_hidden'], config['rpn_hidden'], kernel_size=config['kernel_size'], stride=1, padding=1)
        self.cls  = Conv2d(config['rpn_hidden'], config['k']*2, kernel_size=1, stride=1, padding=0)
        self.bbox = Conv2d(config['rpn_hidden'], config['k']*4, kernel_size=1, stride=1, padding=0)
        self.relu = ReLU()
        for m in [self.bbox, self.cls, self.conv]:
            init_weights.normal_(m.weight, mean=config['rpn_mean'], std=config['rpn_std'])
            init_weights.constant_(m.bias, 0)

    def forward(self, x):
        B = x.size(0)
        x = self.conv(x)
        x = self.relu(x)

        cls_logits = self.cls(x)
        cls_logits = cls_logits.permute(0,2,3,1).reshape(B, -1, 2)
        cls_logits = cls_logits - cls_logits.max(dim=2, keepdim=True)[0]

        #cls_logits = cls_logits.permute(0,2,3,1).contiguous().view(B, -1, 2)
        cls_softmax = F.softmax(cls_logits, dim=2)

        # dx,dy,dw,dh
        bbox_reg = self.bbox(x)
        bbox_reg = bbox_reg.permute(0,2,3,1).reshape(B, -1, 4)
        #bbox_reg = bbox_reg.permute(0,2,3,1).contiguous().view(B, -1, 4)

        return cls_logits, cls_softmax, bbox_reg

class RoI_MiniBatch_Sampler(Module):
    def __init__(self, 
        R=config['mini_R'], 
        pos_ratio=config['mini_pos_ratio'],
        pos_th=config['mini_pos_th'],
        neg_lo=config['mini_neg_lo'],
        neg_hi=config['mini_neg_hi'],
    ):
        super().__init__()
        self.R = R
        self.pos_ratio = pos_ratio
        self.pos_th = pos_th
        self.neg_lo = neg_lo
        self.neg_hi = neg_hi

    def forward(self, inputs, targets):
        with torch.no_grad():
            images, all_rois = inputs
            bboxes = targets['bboxes']
            classes = targets['class_ids']
            # minibatch sampl
            batch_size = len(images)
            N = self.R // batch_size

            target_class = []
            target_bbox = []
            rois = []

            for i in range(batch_size):
                im = images[i]
                im_rois = all_rois[i]
                im_bboxes = bboxes[i]
                im_class = classes[i]

                # sampling during training
                im_bboxes, im_class, pos_mask, neg_mask, _ = U.label_rois(im_rois, im_bboxes, im_class, pos_lo=self.pos_th, neg_lo=self.neg_lo, neg_hi=self.neg_hi)
                n_pos = pos_mask.sum()
                n_neg = neg_mask.sum()
                n_rois = im_rois.size(0)

                pos_sample_size = min(n_pos, int(N * self.pos_ratio))
                neg_sample_size = min(n_neg, N - pos_sample_size)

                indices = torch.arange(n_rois).to(im.device)
                pos_idx = U.rand_sample(indices[pos_mask], pos_sample_size)[1]
                neg_idx = U.rand_sample(indices[neg_mask], neg_sample_size)[1]
                sampled_idx = torch.cat([pos_idx, neg_idx])
                im_class[neg_idx] = 0
                im_class = im_class[sampled_idx]
                im_bboxes = im_bboxes[sampled_idx]
                im_rois = im_rois[sampled_idx]
                # sampling during training

                rois.append(im_rois)
                target_bbox.append(im_bboxes)
                target_class.append(im_class)

            target_class = torch.cat(target_class)
            target_bbox = torch.cat(target_bbox)
            rois = torch.cat(rois)

            return images, rois, target_class, target_bbox

class MultiTaskLoss(Module):
    # https://arxiv.org/pdf/1504.08083
    # FastRCNN, multi-task loss section
    def __init__(self, lam):
        super().__init__()
        self.lam = lam
        #self.lam = 1
        self.classification_loss = CrossEntropyLoss(reduction='sum')
        self.localization_loss = SmoothL1Loss(reduction='sum', beta=1.0)

    def forward(self, cls_logits, cls_target, reg_prediction, reg_target, Ncls=1, Nreg=1):
        '''
            L(p,u,tu,v)
            p -> softmax over K+1 classes, 0 = background
              -> or logits, if cross entropy loss is used
            u -> ground truth class
            tu -> predicted transformation (bbox regression prediction)
            v -> regression target
            = Lcls + lam * [u >= 1] Lloc

            normalization terms, form fasterRCNN v3
            N_cls = # of sampled ROIS
            N_reg = # of anchors (??)
            loss = 1/N_cls * L_cls + lam * 1/N_reg * L_reg
        '''
        reg_mask = (cls_target >= 1)
        l_cls = 1/Ncls * self.classification_loss(cls_logits, cls_target)
        if reg_mask.any():
            l_loc = 1/Nreg * self.localization_loss(reg_prediction[reg_mask], reg_target[reg_mask])
        else:
            l_loc = 0.0
        loss = l_cls + self.lam * l_loc
        return loss


class FasterRCNN(Module):
    # may be broken
    def __init__(self, 
        step=None,
        config=config,
    ):
        super().__init__()
        self.step = step
        self.config = config
        backbone = Backbone()
        self.features = backbone.features # trainable
        self.detection_layer = DetectionHead(
            backbone.classifier,
            config=config,
        ) # trainable
        self.rpn_layer = RPNHead(
            config=config,
        ) # trainable
        self.anchor_layer = AnchorGenerator(
            config=config,
        )
        self.merge_layer = TransformBBox()
        self.minibatch = RoI_MiniBatch_Sampler(
            R=config['mini_R'],
            pos_ratio=config['mini_pos_ratio'],
            pos_th=config['mini_pos_th'],
            neg_lo=config['mini_neg_lo'],
            neg_hi=config['mini_neg_hi'],
        )
        self.nms_th = config['nms_th']

    def forward(self, inputs, targets=None):
        outputs = {}
        images = inputs[0]
        assert len(images) == 1

        x = images[0].unsqueeze(0) #( 1, 3, im_h, im_w)
        ft = self.features(x) #(1, C, H, W)

        # RPN head
        cls_logits, cls_softmax, bbox_reg = self.rpn_layer(ft) # (1, C*H*W, 2), ..., #(1, C*H*W, 4)
        cls_logits = cls_logits[0]
        cls_softmax = cls_softmax[0]
        bbox_reg = bbox_reg[0]

        anchors = self.anchor_layer(ft.size(2), ft.size(3))
        anchors = anchors.to(ft.device)
        roi_proposals = self.merge_layer(anchors.unsqueeze(0), bbox_reg.unsqueeze(0))
        roi_proposals = roi_proposals.squeeze()

        if self.training:
            # drop cross boundary
            kept = U.drop_cross_boundary_boxes(anchors, x.size(3), x.size(2)).to(ft.device) # reduces anchors from 20k to 6k
            cls_logits = cls_logits[kept]
            cls_softmax = cls_softmax[kept]
            bbox_reg = bbox_reg[kept]
            anchors = anchors[kept]
            roi_proposals = roi_proposals[kept]
        else:
            # apply fully convolutional RPN;
            # clip proposals
            roi_proposals = U.clip_bboxes(roi_proposals, x.size(3), x.size(2)).to(x.device)

            # drop zero width/height
            rxywh = U.xyxy_2_xywh(roi_proposals)
            rw = rxywh[:, 2]
            rh = rxywh[:, 3]
            kept_mask = (rw > 0) & (rh > 0)
            cls_logits = cls_logits[kept_mask]
            cls_softmax = cls_softmax[kept_mask]
            bbox_reg = bbox_reg[kept_mask]
            anchors = anchors[kept_mask]
            roi_proposals = roi_proposals[kept_mask]

        # NMS is applied leaving 2000 proposals
        # Fast RCNN is trained using the 2000 proposals
        score = cls_softmax[:, 1]
        kept = nms(roi_proposals, score, self.nms_th)
        kept = kept[:self.config['n_proposals']] # N proposals
        cls_logits = cls_logits[kept]
        cls_softmax = cls_softmax[kept]
        bbox_reg = bbox_reg[kept]
        anchors = anchors[kept]
        roi_proposals = roi_proposals[kept]

        if self.training:
            # label anchors
            gt_bboxes, gt_class, pos_mask, neg_mask, ious = U.label_rois(
                anchors,
                targets['bboxes'][0],
                targets['class_ids'][0],
                pos_lo=self.config['rpn_sample_hi'],
                neg_hi=self.config['rpn_sample_lo']
            )
            # take highest iou to be fg
            anchor_idx_per_gt = ious.argmax(dim=0)
            pos_mask[anchor_idx_per_gt] = True
            neg_mask[anchor_idx_per_gt] = False
            gt_class[neg_mask] = 0
            gt_class[pos_mask] = 1

            # sample up to 50% of 256 to be positive
            indices = torch.arange(anchors.size(0)).to(anchors.device)
            pos_sample_size = min(pos_mask.sum(), self.config['rpn_sample_size']//2)
            neg_sample_size = min(neg_mask.sum(), self.config['rpn_sample_size'] - pos_sample_size)
            pos_idx = U.rand_sample(indices[pos_mask], pos_sample_size)[1]
            neg_idx = U.rand_sample(indices[neg_mask], neg_sample_size)[1]
            sampled_idx = torch.cat([pos_idx, neg_idx])

            outputs['rpn_iou'] = ious.max(dim=1)[0][sampled_idx]
            outputs['rpn_gtbboxes'] = gt_bboxes[sampled_idx]
            outputs['rpn_roi'] = roi_proposals[sampled_idx]
            outputs['rpn_cls_pred'] = cls_softmax.argmax(dim=1)[sampled_idx]
            outputs['rpn_cls_softmax'] = cls_softmax[sampled_idx]
            outputs['rpn_cls_logits'] = cls_logits[sampled_idx]
            outputs['rpn_cls_target'] = gt_class[sampled_idx]
            outputs['rpn_bbox_reg'] = bbox_reg[sampled_idx]
            outputs['rpn_reg_target'] = U.parameterize_bbox(gt_bboxes[sampled_idx], roi_proposals[sampled_idx])
            outputs['rpn_anchors'] = anchors[sampled_idx]

        else:
            outputs['rpn_roi'] = roi_proposals
            outputs['rpn_cls_softmax'] = cls_softmax
            outputs['rpn_cls_logits'] = cls_logits
            outputs['rpn_cls_pred'] = cls_softmax.argmax(dim=1)
            outputs['rpn_bbox_reg'] = bbox_reg
            outputs['rpn_anchors'] = anchors

        if inputs[1] is None:
            rois = [roi_proposals]
        else:
            rois = inputs[1]
        roi_proposals = rois[0]

        if self.training:
            # sample 128 for training, 25% of which are positive
            images, roi_proposals, target_class, target_bbox = self.minibatch((images, [roi_proposals]), targets)
            targets['gt_bboxes'] = target_bbox
            targets['gt_class'] = target_class

        n_proposals = roi_proposals.size(0)
        # Detection Head
        pooled_fts = self.detection_layer.roi_pool(ft, [roi_proposals])
        pooled_fts = pooled_fts.contiguous().view(roi_proposals.size(0), -1)
        cls_logits, cls_softmax, bbox_reg = self.detection_layer(pooled_fts)
        cls_pred = cls_softmax.argmax(dim=1)

        indices = torch.arange(n_proposals)
        if self.training or targets:
            # (N, K, 4) -> (N, 4)
            bbox_reg = bbox_reg[indices, cls_pred]
            bbox_pred = self.merge_layer(roi_proposals.unsqueeze(0), bbox_reg.unsqueeze(0))
            bbox_pred = bbox_pred.squeeze()
            outputs['det_cls_pred'] = cls_pred
            outputs['det_bbox_pred'] = bbox_pred
            outputs['det_rois'] = roi_proposals

            outputs['det_bbox_reg'] = bbox_reg
            outputs['det_reg_target'] = U.parameterize_bbox(targets['gt_bboxes'], roi_proposals)
            outputs['det_cls_logits'] = cls_logits
            outputs['det_cls_target'] = targets['gt_class']

        else:
            # NMS applied for each non-bg class indepedently
            per_class = []
            for i in range(1, cls_softmax.size(1)):
                pred_cls = torch.full((n_proposals,), i)
                pred_cls = pred_cls.to(x.device)
                _logits = cls_logits[indices, i]
                _scores = cls_softmax[indices, i]
                _bbox_reg = bbox_reg[indices, i]
                _bbox_pred = self.merge_layer(roi_proposals.unsqueeze(0), _bbox_reg.unsqueeze(0))
                _bbox_pred = _bbox_pred.squeeze()

                # clip predicted bbox to to image size
                _bbox_pred = U.clip_bboxes(_bbox_pred, x.size(3), x.size(2)).to(x.device)

                # keep non zero-width/height
                xywh = U.xyxy_2_xywh(_bbox_pred)
                keep = (xywh[:,2] > 0) & (xywh[:,3] > 0)
                _bbox_pred = _bbox_pred[keep]
                _logits = _logits[keep]
                _scores = _scores[keep]
                pred_cls = pred_cls[keep]

                # NMS
                kept = nms(_bbox_pred, _scores, self.nms_th)

                per_class.append([
                    pred_cls[kept],
                    _scores[kept],
                    _bbox_pred[kept],
                ])

            pred_cls, scores, bbox_pred = zip(*per_class)

            pred_cls = torch.cat(pred_cls)
            scores = torch.cat(scores)
            bbox_pred = torch.cat(bbox_pred)

            outputs['det_cls_pred'] = pred_cls
            outputs['det_bbox_pred'] = bbox_pred
            outputs['det_score'] = scores

        return outputs, targets

class FasterRCNN_RPNLoss(MultiTaskLoss):
    def __init__(self, config=config):
        lam = config['rpn_loss_lambda']
        super().__init__(lam=lam)
        self.config = config

    def forward(self, outputs, y):
        cls_logits = outputs['rpn_cls_logits']
        cls_tgt = outputs['rpn_cls_target']
        bbox_reg = outputs['rpn_bbox_reg']
        bbox_tgt = outputs['rpn_reg_target']

        nreg = torch.where(cls_tgt >= 1, 1, 0).sum()
        ncls = cls_logits.size(0)
        loss = super().forward(cls_logits, cls_tgt, bbox_reg, bbox_tgt, Ncls=ncls, Nreg=nreg)
        return loss

class FasterRCNN_FastLoss(MultiTaskLoss):
    def __init__(self, config=config):
        super().__init__(lam=config['fast_loss_lambda'])
        self.config = config

    def forward(self, outputs, y):
        cls_logits = outputs['det_cls_logits']
        cls_tgt = outputs['det_cls_target']
        bbox_reg = outputs['det_bbox_reg']
        bbox_tgt = outputs['det_reg_target']

        nreg = torch.where(cls_tgt >= 1, 1, 0).sum()
        ncls = cls_logits.size(0)
        loss = super().forward(cls_logits, cls_tgt, bbox_reg, bbox_tgt, Ncls=ncls, Nreg=nreg)
        return loss

