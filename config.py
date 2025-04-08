config = {
    'stride_len': 16,
    'scales': [128,256,512],
    'ratios': [(1,1),(1,2),(2,1)],

    'kernel_size': 3,
    'rpn_hidden': 512,
    'rpn_mean': 0.0,
    'rpn_std': 0.01,

    'pool_W': 7,
    'pool_H': 7,
    'det_hidden': 4096,
    'det_classes': 21,
    'det_mean': 0.0,
    'det_std_cls': 0.01,
    'det_std_box': 0.001,
    'det_bias': 0.0,

    'mini_R': 128,
    'mini_pos_ratio': .25,
    'mini_pos_th': .5,
    'mini_neg_lo': 0,
    'mini_neg_hi': .5,

    'rpn_loss_lambda': 10.0,
    'rpn_sample_lo': .3,
    'rpn_sample_hi': .7,
    'rpn_sample_size': 256,

    'fast_loss_lambda': 1.0,

    'rpn_lr_0': 0.001,
    'rpn_lr_1': 0.0001,
    'rpn_step0': 60000,
    'rpn_step1': 20000,

    'fast_lr_0': 0.001,
    'fast_lr_1': 0.0001,
    'fast_step0': 30000,
    'fast_step1': 10000,

    'sgd_decay': 5e-4,
    'sgd_momentum': .9,

    'nms_th': 0.7,
}

config['areas'] = list(map(lambda x: x**2, config['scales']))
config['k'] = len(config['ratios']) * len(config['areas'])

# filepaths
roi_proposal_path = './outputs/roi_proposals.pkl'
checkpoint_filename_template_1 = './outputs/checkpoint_step1_{step}.pt'
checkpoint_filename_template_2 = './outputs/checkpoint_step2_{step}.pt'
checkpoint_filename_template_3 = './outputs/checkpoint_step3_{step}.pt'
checkpoint_filename_template_4 = './outputs/checkpoint_step4_{step}.pt'
