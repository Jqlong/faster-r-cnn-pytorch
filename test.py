import copy
import os

import xmltodict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.io.image import read_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    faster_rcnn,
)
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as transforms

from utils import parse_xml


def resize_bbox(bbox, w, h):
    """
    bbox: (xmin, ymin, xmax, ymax)
    """
    scaled_bbox = copy.deepcopy(bbox)
    scaled_bbox[:, 0] *= w / 224
    scaled_bbox[:, 1] *= h / 224
    scaled_bbox[:, 2] *= w / 224
    scaled_bbox[:, 3] *= h / 224
    return scaled_bbox


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the model
model = fasterrcnn_resnet50_fpn_v2(box_score_thresh=0.3)

# modify the model to fit the number of classes, the number of classes is 2, background and car
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 2
model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model_path = "model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# load the image and transform it
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

test_img_path = "D:\\Dataset\\PASCAL_VOC_2007\\voc_car\\train\\JPEGImages\\"
test_xml_path = "D:\\Dataset\\PASCAL_VOC_2007\\voc_car\\train\\Annotations\\"
output_dir = './output_images'
os.makedirs(output_dir, exist_ok=True)

for img_file in os.listdir(test_img_path):
    img_path = os.path.join(test_img_path, img_file)
    # print(img_path)
    xml_path = os.path.join(test_img_path.replace("JPEGImages", "Annotations"), img_file.replace(".jpg", ".xml"))
    # print(xml_path)
    img = read_image(img_path)
    h, w = img.shape[1], img.shape[2]
    img_input = transform(img).to(device).reshape(1, 3, 224, 224)
    bbox = parse_xml(xml_path)
    bbox = torch.as_tensor(bbox, dtype=torch.float32)

    model.eval()
    img_show = copy.deepcopy(img)
    img_show = torch.as_tensor(img_show, dtype=torch.uint8)

    with torch.no_grad():
        output = model(img_input)
        # resize the bounding box to the original size
        bbox_show = resize_bbox(output[0]["boxes"], w, h)
        # convert the label (1 -> car)
        labels = []
        for i in range(len(output[0]["labels"])):
            if output[0]["labels"][i] == 1:
                labels.append("car")
            else:
                labels.append("background")
        img = draw_bounding_boxes(img_show, bbox_show, colors=(0, 255, 0), labels=labels, width=2)

    output_img_path = os.path.join(output_dir, img_file)
    plt.imsave(output_img_path, img.permute(1, 2, 0).numpy())
    print(f"Processed and saved: {output_img_path}")
    # plt.imshow(img.permute(1, 2, 0))


# img = read_image(test_img_path)
# h, w = img.shape[1], img.shape[2]
# img_input = transform(img).to(device).reshape(1, 3, 224, 224)
# bbox = parse_xml(test_xml_path)
# bbox = torch.as_tensor(bbox, dtype=torch.float32)
#
# # inference and draw the bounding box
# model.eval()
# img_show = copy.deepcopy(img)
# img_show = torch.as_tensor(img_show, dtype=torch.uint8)
#
# with torch.no_grad():
#     output = model(img_input)
#     # resize the bounding box to the original size
#     bbox_show = resize_bbox(output[0]["boxes"], w, h)
#     # convert the label (1 -> car)
#     labels = []
#     for i in range(len(output[0]["labels"])):
#         if output[0]["labels"][i] == 1:
#             labels.append("car")
#         else:
#             labels.append("background")
#     img = draw_bounding_boxes(img_show, bbox_show, colors=(0, 255, 0), labels=labels, width=2)
#
# plt.imshow(img.permute(1, 2, 0))
# plt.show()
