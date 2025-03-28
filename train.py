from dataset import VOCDataset
from torch.utils.data import DataLoader
from datatypes import (
	Rpn_cfg,
	Detection_cfg,
	Anchor_cfg,
)
from models import MultiTaskLoss, RPNLoss, freeze_module
from torch.optim.lr_scheduler import LambdaLR
from itertools import cycle
import utils as U
from torch.optim import (
  SGD
)
from torchvision.ops import (
    nms
)
import torch

stride_len = 16
scales = [128,256,512]
areas = list(map(lambda x: x**2, scales))
ratios = [(1,1), (1,2), (2,1)]
k = len(areas) * len(ratios)

cfg_1 = Rpn_cfg(3, 512, k, 0.0, 0.01)
cfg_2 = Detection_cfg(7, 7, 1/stride_len, 4096, 21, 0.01, 0.001, 0.0)
cfg_3 = Anchor_cfg(areas, ratios, stride_len, .3, .7, 256)

def collate_fn(batch):
	return batch

def get_lr_scheduler(optimizer, milestone, lr_1, lr_2):
  def lambda_lr(step):
    if step < milestone:
      lr = lr_1
    else:
      lr = lr_2
    return lr

  return LambdaLR(optimizer, lr_lambda=lambda_lr)

def train_rpn_net(model, dataset, steps=60000+20000, **kwargs):
	dataloader = DataLoader(dataset, shuffle=True, batch_size=1, collate_fn=collate_fn)
	optimizer = SGD(model.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)
	scheduler = get_lr_scheduler(optimizer, 60000, 0.001, 0.0001)
	criterion = RPNLoss(10)
	dataloader = cycle(dataloader)
	device = kwargs['device']

	model = model.to(device)

	for step in range(steps):
		batch = next(dataloader)
		sample = batch[0]

		# zero gradients
		optimizer.zero_grad()
		gt = sample['bboxes'].to(device)
		img = sample['image'].unsqueeze(0).to(device)
		outputs = model(img)
		cls_logits = outputs.cls_logits
		anchors = outputs.anchors
		roi_proposals = outputs.roi_proposals

		sampled_anchors, sampled_indices, sampled_labels, sampled_gt_idx = U.sample_anchors(
			anchors, gt, .3, .7, 256
		)
		# objectness labels for sampled
		gt_labels = sampled_labels
		pred_labels = cls_logits.squeeze()[sampled_indices]

		# roi for sampled
		pred_box = roi_proposals[sampled_indices]
		gt_box = gt[sampled_gt_idx]
		# parameterized as per paper
		pred_box = U.parameterize_bbox(pred_box, sampled_anchors)
		gt_box = U.parameterize_bbox(gt_box, sampled_anchors)

		# loss
		loss = criterion(pred_labels, gt_labels, pred_box, gt_box)
		print(step, loss.item())

		loss.backward()
		optimizer.step()
		scheduler.step()


	return model

def generate_rpn_proposals(model, dataset, **kwargs):
	device = kwargs['device']
	model = model.to(device)
	model.eval()
	dataloader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)

	all = {}
	with torch.no_grad():
		for i, samples in enumerate(dataloader):
			if kwargs['breakpoint'] and kwargs['breakpoint'] == i:
				break
	
			sample = samples[0]
			img = sample['image'].unsqueeze(0).to(device)
			im_h = img.size(2)
			im_w = img.size(3)
	
			outputs = model(img)
			cls_scores = outputs.cls_softmax.squeeze()[:, 1]
			anchors = outputs.anchors
			roi_proposals = outputs.roi_proposals
	
			# clamp cross-boundary rois
			roi_proposals = U.clip_bboxes(roi_proposals, im_h, im_w)
	
			# drop 0 width or height rois
			roi_xywh = U.xyxy_2_xywh(roi_proposals)
			condition = (roi_xywh[:, 2] > 0) & (roi_xywh[:, 3] > 0)
			roi_proposals = roi_proposals[condition]
			cls_scores = cls_scores[condition]
	
			# non-maximum supression
			kept_idx = nms(roi_proposals, cls_scores, kwargs['nms_th'])
			# keep only top-N by scores
			kept_idx = kept_idx[:kwargs['topN']]
	
			cls_scores = cls_scores[kept_idx]
			roi_proposals = roi_proposals[kept_idx]

			all[sample['filename']] = roi_proposals.cpu()

	return all

'''
SGD hyper-parameters. The fully connected layers used
for softmax classification and bounding-box regression are
initialized from zero-mean Gaussian distributions with standard deviations 0.01 and 0.001, 
respectively. Biases are initialized to 0. All layers use a per-layer learning rate of 1 for
weights and 2 for biases and a global learning rate of 0.001.
When training on VOC07 or VOC12 trainval we run SGD
for 30k mini-batch iterations, and then lower the learning
rate to 0.0001 and train for another 10k iterations. When
we train on larger datasets, we run SGD for more iterations,
as described later. A momentum of 0.9 and parameter decay
of 0.0005 (on weights and biases) are used.


 During fine-tuning, each SGD
mini-batch is constructed from N = 2 images, chosen uniformly at random 
(as is common practice, we actually iterate over permutations of the dataset). We use mini-batches
of size R = 128, sampling 64 RoIs from each image. As
in [9], we take 25% of the RoIs from object proposals that
have intersection over union (IoU) overlap with a groundtruth bounding box of at least 0.5. These RoIs comprise
the examples labeled with a foreground object class, i.e.
u ≥ 1. The remaining RoIs are sampled from object proposals that have a maximum IoU with ground truth in the interval [0.1, 0.5), following [11]. These are the background
examples and are labeled with u = 0. The lower threshold
of 0.1 appears to act as a heuristic for hard example mining
[8]. During training, images are horizontally flipped with
probability 0.5. No other data augmentation is used
'''

def train_fastrcnn_net(model, dataset, steps=30000+10000, **kwargs):
	device = kwargs['device']
	model = model.to(device)
	R = 128
	N = 2
	per_image = R//N
	optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
	scheduler = get_lr_scheduler(optimizer, 30000, 0.001, 0.0001)
	dataloader = DataLoader(dataset, shuffle=kwargs['shuffle'], batch_size=N, drop_last=True, collate_fn=collate_fn)
	dataloader = cycle(dataloader)
	criterion = MultiTaskLoss(1)

	for step in range(steps):
		# prepare minibatch
		batch = next(dataloader)
		optimizer.zero_grad()

		accumulated_loss = 0
		for i, sample in enumerate(batch):
			gt_bbox = sample['bboxes']
			roi_proposals = sample['roi_proposals']

			sampled_rois, sampled_indices, sampled_labels, sampled_gt_idx = U.sample_rois(roi_proposals, gt_bbox, 0.5, 0.1, 0.5, R//N, .25)
			# assign 0 to negative rois, assign class id to other rois
			gt_idx = sampled_gt_idx[sampled_labels==1]
			gt_labels = 1 + sample['class_ids'][gt_idx]
			sampled_labels[sampled_labels == 1] = gt_labels

			sample_size = sampled_rois.size(0)
			img_indices = torch.full((sample_size,), 0)

			rois = torch.cat([
				img_indices.unsqueeze(1),
				sampled_rois,
			], dim=1)
			rois = rois.to(device)

			img = sample['image'].unsqueeze(0)
			img = img.to(device)

			outs = model(img, rois)
			cls_logits = outs.cls_logits
			cls_softmax = outs.cls_softmax

			bbox_reg = outs.bbox_reg
			pred_locs = outs.locs.cpu()
			pred_cls = outs.cls.cpu()

			is_correct = (pred_cls == sampled_labels) & (sampled_labels > 0)
			pred_correct_gt_idx = sampled_gt_idx[is_correct]
			pred_correct_gt_bbox = gt_bbox[pred_correct_gt_idx]
			pred_correct_locs = pred_locs[is_correct]
			pred_correct_rois = sampled_rois[is_correct]
			pred_labels = cls_logits.cpu()
			gt_labels = sampled_labels

			t_pred = U.parameterize_bbox(pred_correct_locs, pred_correct_rois)
			t_gt   = U.parameterize_bbox(pred_correct_gt_bbox, pred_correct_rois)
			loss = criterion(pred_labels, gt_labels, t_pred, t_gt)
			accumulated_loss += loss.item()
			loss.backward()

		print(step, accumulated_loss/2)
		optimizer.step()
		scheduler.step()

def train_shared_rpn(model, dataset, steps=60000+20000, **kwargs):
	device = kwargs['device']
	model = model.to(device)

	dataloader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)
	dataloader = cycle(dataloader)
	optimizer = SGD(model.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)
	scheduler = get_lr_scheduler(optimizer, 60000, 0.001, 0.0001)
	criterion = RPNLoss(10)

	for step in range(steps):
		criterion.zero_grad()
		batch = next(dataloader)
		sample = batch[0]
		gt = sample['bboxes'].to(device)
		img = sample['image'].unsqueeze(0)
		img = img.to(device)

		outputs = model(img)
		outputs['rpn']
		anchors = outputs['anchors']
		roi_proposals = outputs['roi_proposals']

		cls_logits = outputs['rpn'].cls_logits

		sampled_anchors, sampled_indices, sampled_labels, sampled_gt_idx = U.sample_anchors(
			anchors, gt, .3, .7, 256
		)
		# objectness labels for sampled
		gt_labels = sampled_labels
		pred_labels = cls_logits.squeeze()[sampled_indices]

		# roi for sampled
		pred_box = roi_proposals[sampled_indices]
		gt_box = gt[sampled_gt_idx]
		# parameterized as per paper
		pred_box = U.parameterize_bbox(pred_box, sampled_anchors)
		gt_box = U.parameterize_bbox(gt_box, sampled_anchors)

		loss = criterion(pred_labels, gt_labels, pred_box, gt_box)
		print(step, loss.item())

		loss.backward()
		optimizer.step()
		scheduler.step()


