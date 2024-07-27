import json
import numpy as np
from PIL import Image
import torch
import cv2
import os
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SegmentationDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        with open(os.path.join("./data/", annotations_file)) as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        img_info = self.data['images'][idx]
        img_path = os.path.join("./data/", img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        mask = np.zeros((image.height, image.width), dtype=np.float32)
        image = np.array(image)

        for ann in self.data['annotations']:
            if ann['image_id'] == img_info['id']:
                for seg in ann['segmentation']:
                    seg = [seg[i:i+2] for i in range(0, len(seg), 2)]
                    seg_np = np.array(seg, dtype=np.int32)
                    rr, cc = seg_np[:, 1], seg_np[:, 0]
                    rr = np.clip(rr, 0, mask.shape[0] - 1)
                    cc = np.clip(cc, 0, mask.shape[1] - 1)
                    mask[rr, cc] = 1.0

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
