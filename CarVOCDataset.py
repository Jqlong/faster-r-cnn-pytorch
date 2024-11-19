import os

import cv2

import torch

from utils import parse_xml


class CustomCarVOCDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=None, split="train"):
        self.root_dir = root_dir  # 数据集路径
        self.transforms = transforms
        self.split = split
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, split, "JPEGImages"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root_dir, split, "Annotations"))))
        # print("len of imgs: ", len(self.imgs), "len of annotations: ", len(self.annotations))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "train" if self.split == "train" else "val", "JPEGImages", self.imgs[idx])
        img = cv2.imread(img_path)

        # 读取xml文件
        xml_path = os.path.join(self.root_dir, "train" if self.split == "train" else "val", "Annotations", self.annotations[idx])
        bndboxs = parse_xml(xml_path)  # 获取每个的边界框

        num_objs = len(bndboxs)  # 多少个对象
        boxes = bndboxs
        labels = torch.ones((num_objs), dtype=torch.int64)  # 标签全都为1
        # print(boxes)
        # print(labels)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # print(target["boxes"])

        if self.transforms is not None:
            # 进行图像增强
            h, w = img.shape[: 2]  # 获取图像的高和宽度
            # print(h, w)
            _h, _w = 224, 224
            target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
            target["boxes"][:, 0] *= _w / w
            target["boxes"][:, 1] *= _h / h
            target["boxes"][:, 2] *= _w / w
            target["boxes"][:, 3] *= _h / h  # 将边界框的每个坐标调整为 224 224 匹配的大小
            # print(target["boxes"])
            img = self.transforms(img)

        is_crowd = torch.zeros((num_objs), dtype=torch.int64)  # 群体，默认为0
        target["iscrowd"] = is_crowd

        area = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])  # 计算边框面积
        target["area"] = area

        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        return img, target

    def __len__(self):
        return len(self.imgs)





# if __name__ == "__main__":
#     root_dir = 'D:\\Dataset\\PASCAL_VOC_2007\\voc_car\\'
#     img = CarVOCDataset(root_dir).__getitem__(2)
