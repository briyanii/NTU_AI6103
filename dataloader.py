import torch
from torchvision.transforms import v2
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
import torchvision
import torch.nn.functional as F
from torchvision.io import read_image
from itertools import islice
import pickle
import utils as U

def get_dataloader(
    dataset,
    num_samples=1000,
    seed=None,
    batch_size=1,
    drop_last=True,
    skip=0,
    augment_th=.5,
    normalize=True,
    augment=False,
    shuffle=True,
    roi_proposal_path=None,
    num_workers=0,
    prefetch_factor=None,
):
    rng = torch.Generator()
    if seed:
        rng.manual_seed(seed)

    proposals = []
    if roi_proposal_path:
        with open(roi_proposal_path, 'rb') as fp:
            proposals = pickle.load(fp)
    if shuffle:
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples, generator=rng)
    else:
        sampler = SequentialSampler(dataset)

    do_augment = torch.rand(num_samples, generator=rng) > augment_th
    do_augment = do_augment & augment

    transform = []
    transform.append(v2.ToDtype(torch.float32, scale=True))
    if normalize:
        transform.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform = v2.Compose(transform)

    flip = v2.RandomHorizontalFlip(p=1)

    def collate_fn(batch):
        imgs = []
        rois = [] if proposals else None
        boxes = []
        indices = []
        cls = []
        is_aug = []
        widths = []
        heights = []

        max_w = 0
        max_h = 0
        for i, (idx, aug) in enumerate(batch):
            is_aug.append(aug.item())
            aug = aug.item()
            item = dataset[idx]

            ### process image
            w = item['width']
            h = item['height']
            min_side = min(w,h)
            h = h * 600 / min_side
            w = w * 600 / min_side
            max_side = max(w,h)
            if max_side > 1000:
                h = h * 1000 / max_side
                w = w * 1000 / max_side

            scaling_factor = w / item['width']
            w = int(w)
            h = int(h)
            widths.append(w)
            heights.append(h)

            max_w = max(w, max_w)
            max_h = max(h, max_h)

            img = read_image(item['filepath'])
            img = v2.Resize((h,w), antialias=True)(img)
            img = transform(img)
            if aug:
                img = flip(img)
            imgs.append(img)

            box = torch.Tensor(item['bboxes']) * scaling_factor
            box = box.floor().to(torch.float32)

            cids = torch.Tensor(item['class_ids'])
            cids = cids.to(torch.int64)

            if aug:
                x1,y1,x2,y2 = box.unbind(dim=1)
                box = torch.stack([w-x2,y1,w-x1,y2], dim=1)

            if proposals:
                roi = proposals[idx]
                roi = roi[:, 1:]
                x1,y1,x2,y2 = roi.unbind(dim=1)
                cond = (x1 < x2) & (y1 < y2) & (0 <= x1) & (0 <= y1)
                if aug:
                    roi = torch.stack([w-x2,y1,w-x1,y2], dim=1)
                roi = U.cat_val(roi, i)
                rois.append(roi)

            cls.append(cids)
            boxes.append(box)

        batch = {
            'x': (imgs, rois), # list[ (C, H, W) ], list[ (K, 5) ]
            'y': {
                'width': widths, # list
                'height': heights, # list
                'bboxes': boxes, # list[ (M,) ]
                'class_ids': cls, # list[ (M,) ]
                'is_augmented': is_aug, # list
            }
        }

        return batch

    ls = list(zip(sampler, do_augment))
    dataloader = DataLoader(ls, batch_size=batch_size, drop_last=drop_last, num_workers=num_workers, prefetch_factor=prefetch_factor, collate_fn=collate_fn)

    dataloader = islice(dataloader, skip, None)
    return dataloader, rng.initial_seed()
