from datatypes import (
	Rpn_cfg,
	Detection_cfg,
	Anchor_cfg,
)

from dataset import VOCDataset

stride_len = 16
scales = [128,256,512]
areas = list(map(lambda x: x**2, scales))
ratios = [(1,1), (1,2), (2,1)]
k = len(areas) * len(ratios)

config = {
    'stride_len': 16,
    'scales': [128,256,512],
    'ratios': [(1,1),(1,2),(2,1)],
    'kernel_size': 3,
    'rpn_hidden': 512,
    'rpn_mean': 0.0,
    'rpn_std': 0.01,
    'W': 7,
    'H': 7,
    'det_hidden': 4096,
    'det_classes': 21,
    'det_mean': 0.0,
    'det_std_cls': 0.01,
    'det_std_box': 0.001,
    'det_bias': 0.0,
    'an_lo_th': .3,
    'an_hi_th': .7,
    'an_n': 256,
}
config['areas'] = list(map(lambda x: x**2, config['scales']))
config['k'] = len(config['ratios']) * len(config['areas'])

def get_rpn_cfg():
    return Rpn_cfg(
        config['kernel_size'],
        config['rpn_hidden'],
        config['k'],
        config['rpn_mean'],
        config['rpn_std'],
    )

def get_detection_cfg():
    return Detection_cfg(
        config['W'],
        config['H'],
        1.0, #1/config['stride_len'], # 1.0 since bbox is defined at image original scale
        config['det_hidden'],
        config['det_classes'],
        config['det_std_cls'],
        config['det_std_box'],
        config['det_bias'],
    )

def get_anchor_cfg():
    return Anchor_cfg(
        config['areas'],
        config['ratios'],
        config['stride_len'],
        config['an_lo_th'],
        config['an_hi_th'],
        config['an_n'],
    )

# filepaths
roi_proposal_path = './outputs/roi_proposals.pkl'
checkpoint_filename_template_1 = './outputs/checkpoint_step1_{step}.pt'
checkpoint_filename_template_2 = './outputs/checkpoint_step2_{step}.pt'
