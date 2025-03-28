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

cfg_1 = Rpn_cfg(3, 512, k, 0.0, 0.01)
cfg_2 = Detection_cfg(7, 7, 1/stride_len, 4096, 21, 0.01, 0.001, 0.0)
cfg_3 = Anchor_cfg(areas, ratios, stride_len, .3, .7, 256)


