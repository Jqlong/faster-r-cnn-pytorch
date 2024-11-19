import os
import xmltodict
import numpy as np
import torch
from torchvision.io.image import read_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    faster_rcnn,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
import torchvision.transforms as transforms
from tqdm import tqdm

from CarVOCDataset import CustomCarVOCDataset


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    # parse_xml('D:\\Dataset\\PASCAL_VOC_2007\\voc_car\\train\\Annotations\\000012.xml')


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    root_dir = 'D:\\Dataset\\PASCAL_VOC_2007\\voc_car\\'
    train_dir = os.path.join(root_dir, "train")
    train_dataset = CustomCarVOCDataset(root_dir, transform, split="train")
    dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # 创建模型
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.3)

    # 修改类别
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 2  # 只有两类
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 训练模型
    epochs = 10

    best_loss = 1e10
    for i in tqdm(range(epochs)):
        # print("Epoch {}/{}".format(i, epochs))
        # print("-" * 10)
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for images, targets in tqdm(dataloader):
            input_images = images
            input_images = torch.stack(input_images, dim=0)
            inputs = input_images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            outputs = model(inputs, targets)

            loss = sum(loss for loss in outputs.values())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), "model.pth")

