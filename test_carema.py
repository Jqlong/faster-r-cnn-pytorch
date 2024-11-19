import cv2
import copy
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, faster_rcnn
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
import numpy as np


# Resize bounding boxes function
def resize_bbox(bbox, w, h, input_size=224):
    scaled_bbox = copy.deepcopy(bbox)
    scaled_bbox[:, 0] *= w / input_size
    scaled_bbox[:, 1] *= h / input_size
    scaled_bbox[:, 2] *= w / input_size
    scaled_bbox[:, 3] *= h / input_size
    return scaled_bbox


# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and set it to evaluation mode
model = fasterrcnn_resnet50_fpn_v2(box_score_thresh=0.3)
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 2  # Background and car
model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Start video capture
cap = cv2.VideoCapture(0)  # 0 is the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the frame height and width for bounding box resizing
    h, w = frame.shape[:2]

    # Convert the frame to a tensor and normalize it
    img_tensor = F.to_tensor(cv2.resize(frame, (224, 224))).to(device)
    img_tensor = F.normalize(img_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    img_tensor = img_tensor.unsqueeze(0)

    # Run object detection
    with torch.no_grad():
        outputs = model(img_tensor)

    # Resize bounding boxes to match the original frame size
    bbox_show = resize_bbox(outputs[0]["boxes"], w, h)

    # Draw bounding boxes on the frame
    for bbox, label in zip(bbox_show, outputs[0]["labels"]):
        if label == 1:  # Only draw boxes for "car" (label 1)
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "Car", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Camera Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
