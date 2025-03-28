from torchvision.datasets import VOCDetection
import torchvision

VOCDetection(root='./data', year='2007', image_set='test', download=True)
VOCDetection(root='./data', year='2007', image_set='trainval', download=True)
torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)

