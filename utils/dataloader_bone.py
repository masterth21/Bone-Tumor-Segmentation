import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BoneDataset(data.Dataset):
    def __init__(
        self, image_root, gt_root, trainsize, augmentation=False, split="train"
    ):
        self.trainsize = trainsize
        self.augmentation = augmentation
        self.split = split
        # Hỗ trợ các định dạng ảnh X-ray phổ biến
        exts = (".jpg", ".png", ".jpeg")
        self.images = sorted(
            [
                os.path.join(image_root, f)
                for f in os.listdir(image_root)
                if f.lower().endswith(exts)
            ]
        )
        self.gts = sorted(
            [
                os.path.join(gt_root, f)
                for f in os.listdir(gt_root)
                if f.lower().endswith(exts)
            ]
        )
        self.size = len(self.images)

        # Cấu hình Augmentation hoặc Resize thuần túy
        if self.split == "train" and self.augmentation:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Resize(height=self.trainsize, width=self.trainsize),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height=self.trainsize, width=self.trainsize),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def __getitem__(self, index):
        # 1. Đọc ảnh và chuyển sang RGB
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Đọc nhãn Mask đa giác (đã được tạo từ bước preprocess)
        mask = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)

        # 3. Áp dụng biến đổi (Resize về 256x256)
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        # 4. Chuẩn hóa Mask về nhị phân [0, 1]
        mask = (mask > 0.5).float()
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        if self.split == "train":
            return image, mask
        else:
            name = os.path.basename(self.images[index])
            # Trả về kích thước để hỗ trợ tính toán DSC/IoU chính xác
            return image, mask, (image.shape[1], image.shape[2]), name

    def __len__(self):
        return self.size


def get_loader(
    image_root,
    gt_root,
    batchsize,
    trainsize,
    shuffle=True,
    num_workers=4,
    augmentation=False,
    split="train",
):
    dataset = BoneDataset(image_root, gt_root, trainsize, augmentation, split)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
