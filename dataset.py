import json
import numpy as np
from PIL import Image
import cv2
import os
from torch.utils.data import Dataset
class SegmentationDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        with open(os.path.join("./data/", annotations_file)) as f:
            self.data = json.load(f)
        self.transform = transform
        self.num_classes = len(self.data['categories'])

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        img_info = self.data['images'][idx]
        img_path = os.path.join("./data/", img_info['file_name'])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        for ann in self.data['annotations']:
            if ann['image_id'] == img_info['id']:
                category_id = ann['category_id']
                for seg in ann['segmentation']:
                    seg = [seg[i:i+2] for i in range(0, len(seg), 2)]
                    seg_np = np.array(seg, dtype=np.int32)
                    if len(seg_np) > 0:
                        cv2.fillPoly(mask, [seg_np], category_id)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
