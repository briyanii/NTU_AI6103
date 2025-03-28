from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import torchvision
import torch
import os
from models import StandaloneAnchorGenerator
import xml.etree.ElementTree as ET


class VOCDataset(Dataset):
	def parse_bndbox(self, root):
		bbox = [0,0,0,0]
		# 1
		for c in root:
			i = self.bbox_elements.index(c.tag)
			bbox[i] = int(c.text)
		return bbox
	
	def parse_object(self, root):
		name = None
		bbox = None
		for c in root:
			tag = c.tag
			if tag == 'name':
				name = c.text
			elif tag == 'bndbox':
				bbox = self.parse_bndbox(c)
			class_id = self.object_categories.index(name)
		return name, class_id, bbox
	
	def parse_size(self, root):
		return_val = {}
		for c in root:
			return_val[c.tag] = int(c.text)
		return return_val
	
	def parse_tree(self, root):
		return_val = {'bboxes': [], 'names': [],'class_ids': []}
		for c in root:
			tag = c.tag
			if tag == 'size':
				return_val.update(self.parse_size(c))
			elif tag == 'filename':
				return_val['filepath'] = self.image_path.format(filename=c.text)
				return_val['filename'] = c.text
			elif tag == 'object':
				name, class_id, bbox = self.parse_object(c)
				return_val['names'].append(name)
				return_val['class_ids'].append(class_id)
				return_val['bboxes'].append(bbox)
		return return_val
	
	def image_id_to_labels(self, image_id):
		a_path = self.annotation_path.format(image_id=image_id)
		tree = ET.parse(a_path)
		root = tree.getroot()
		annotation = self.parse_tree(root)
		return annotation
	
	def get_metadata(self, image_set='val', year='2007', root='./data'):
		path = self.set_path.format(
			purpose = 'Main',
			image_set=image_set,
			root=root,
			year=year
		)
		with open(path) as fp:
			image_ids = fp.read().strip().split('\n')
			size = len(image_ids)
			labels = list(map(self.image_id_to_labels, image_ids))
			return labels

	def __init__(self, image_set, anchor_cfg=None, load=True, transform=True, root='./data', year='2007', roi_proposals=None):
		super().__init__()
		self.object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
							'bottle', 'bus', 'car', 'cat', 'chair',
							'cow', 'diningtable', 'dog', 'horse',
							'motorbike', 'person', 'pottedplant',
							'sheep', 'sofa', 'train', 'tvmonitor']
		
		self.base_path = os.path.join('{root}', 'VOCdevkit', 'VOC{year}').format(year=year, root=root)
		self.annotation_path = os.path.join(self.base_path, 'Annotations', '{image_id}.xml')
		self.set_path = os.path.join(self.base_path, 'ImageSets', '{purpose}', '{image_set}.txt')
		self.image_path = os.path.join(self.base_path, 'JPEGImages', '{filename}')
		self.bbox_elements = ['xmin', 'ymin', 'xmax', 'ymax']
		self.roi_proposals = roi_proposals
		self.metadata = self.get_metadata(image_set=image_set, year=year, root=root)
		self.do_load = load
		self.do_transform = transform
		self.do_anchors = anchor_cfg is not None
		if self.do_anchors:
			self.anchor_generator = StandaloneAnchorGenerator(anchor_cfg)

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, index):
		item = {}

		metadata = self.metadata[index]
		
		filepath = metadata['filepath']
		
		item['filepath'] = filepath
		item['filename'] = metadata['filename']

		width = metadata['width']
		height = metadata['height']
		min_side = min(width, height)

		scaling_factor = 600 / min_side
		width = int(round(scaling_factor * width))
		height = int(round(scaling_factor * height))
		item['width'] = width
		item['height'] = height

		bboxes = torch.Tensor(metadata['bboxes']) * scaling_factor
		bboxes = bboxes.round().to(torch.float32)
		item['bboxes'] = bboxes
		
		class_ids = metadata['class_ids']
		class_ids = torch.Tensor(class_ids).to(torch.int64)
		item['class_ids'] = class_ids

		if self.roi_proposals is not None:
			filename = item['filename']
			item['roi_proposals'] = self.roi_proposals.get(filename, None)
	
		if self.do_load:
			img = torchvision.io.read_image(filepath)
			item['image'] = img
			
			if self.do_transform:
				transforms = v2.Compose([
					v2.Resize((height, width), antialias=True),
					v2.ToDtype(torch.float32, scale=True),
					v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				])
				img = transforms(img)
				item['image'] = img
			
			if self.do_anchors:
				anchor_output = self.anchor_generator(img)
				item['anchors'] = anchor_output.anchors
	
		return item

