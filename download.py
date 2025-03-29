from torchvision.datasets import VOCDetection
import torchvision


'''
backbones

# WIP
- vgg16

# TODO
- ZF Net

'''
torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)

'''
Based on experiments on Pascal VOC section the training + test dataset combinations are

TRAIN:
- testing on 07 test
TEST:
1. training on 07 trainval
2. training on 07 trainval + 12 trainval
3. training on 07 trainval + 12 trainval + COCO

TRAIN:
- testing on 12 test
TEST:
1. training on 12 trainval
2. training on 07 trainval + test + 12 trainval
3. training on 07 trainval + test + 12 trainval + COCO
'''

VOCDetection(root='./data', year='2007', image_set='trainval', download=True)
VOCDetection(root='./data', year='2007', image_set='test', download=True)
VOCDetection(root='./data', year='2012', image_set='trainval', download=True)
VOCDetection(root='./data', year='2012', image_set='test', download=True)

# TODO: COCO dataset

