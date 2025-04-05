from collections import namedtuple

TripleOutput = namedtuple('TripleOutput', [
    'cls_logits',
    'cls_softmax',
    'bbox_reg',
])

RPNOutput = namedtuple('RPNOutput', [
    'cls_logits',
    'cls_softmax',
    'bbox_reg',
    'anchors',
    'roi_proposals'
])

FastOutput = namedtuple('FastOutput', [
    'cls_logits',
    'cls_softmax',
    'bbox_reg',
    'locs',
    'cls',
    'roi_proposals',
])

Detection_cfg = namedtuple('DetectionConfig', [
    'W',
    'H',
    'scale',
    'hidden',
    'n_classes',
    'init_cls_std',
    'init_bbox_std',
    'init_bias'
])

Rpn_cfg = namedtuple('RPNConfig', [
    'kernel_size',
    'hidden',
    'k',
    'init_mean',
    'init_std',
])

Anchor_cfg = namedtuple('AnchorConfig', [
    'areas',
    'ratios',
    'stride_len',
    'lo_th',
    'hi_th',
    'n'
])
