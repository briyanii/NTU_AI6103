import torch
from torchvision.datasets import VOCDetection
from torchvision import transforms

from PIL import Image, ImageDraw, ImageFont

from models import FasterRCNN

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog', 'horse',
                            'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor']

def visualize_single_inference(model, image, device, save_path, score_threshold=0.6):
    # In the originnal work, only confidence > 0.6 are kept for visualization

    img = image[0]

    obj_list = image[1]['annotation']['object']

    # Convert to tensor
    to_tensor = transforms.ToTensor()
    tensor_img = to_tensor(img)
    tensor_img = tensor_img.unsqueeze(0)
    tensor_img = tensor_img.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model((tensor_img, None))

    score = outputs['det_score'].cpu().tolist()
    pred_box = outputs['det_bbox_pred'].cpu().tolist()
    pred_cls = outputs['det_cls_pred'].cpu().tolist()

    filter_score = score[score > score_threshold]
    filter_pred_box = pred_box[score > score_threshold]
    filter_pred_cls = pred_cls[score > score_threshold]

    img_with_boxes = img.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    font = ImageFont.load_default()

    for idx, box in enumerate(filter_pred_box):
        pred_score = filter_score[idx]
        pred_cls = filter_pred_cls[idx]
        label = f"predicted class: {object_categories[pred_cls - 1]} ({pred_score:.2f})"
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1] - 10), label, fill="red", font=font)

    for obj in obj_list:
        name = obj['name']
        bbox = obj['bndbox']
        box = [float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']), float(bbox['ymax'])]
        label = f"ground truth: {name}"
        draw.rectangle(box, outline="green", width=2)
        draw.text((box[0], box[1] - 10), label, fill="green", font=font)

    img_with_boxes.save(save_path)

    return img_with_boxes

if __name__ == "__main__":

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FasterRCNN()
    model.to(device)
    state = torch.load('./outputs/checkpoint_step4_80000.pt', map_location=device)
    model.load_state_dict(state['model'])

    voc_test = VOCDetection(root='./data', year='2007', image_set='test', download=False)
    for image in voc_test:
        filename = image[1]['annotation']['filename'].split('.')[0]
        # img_with_boxes.show()
        save_path = f"./outputs/detection_results/{filename}_result.jpg"
        img_with_boxes = visualize_single_inference(model, image, device, save_path)
        print(f"Saved image with bounding boxes to {save_path}")

        