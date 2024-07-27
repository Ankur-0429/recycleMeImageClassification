import json
import numpy as np
from PIL import Image
import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SegmentationDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        with open(annotations_file) as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        img_info = self.data['images'][idx]
        img_path = os.path.join('./data/', img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        mask = np.zeros((image.height, image.width), dtype=np.uint8)

        for ann in self.data['annotations']:
            if ann['image_id'] == img_info['id']:
                for seg in ann['segmentation']:
                    seg = [seg[i:i+2] for i in range(0, len(seg), 2)]
                    seg_np = np.array(seg, dtype=np.int32)
                    rr, cc = seg_np[:, 1], seg_np[:, 0]
                    rr = np.clip(rr, 0, mask.shape[0] - 1)
                    cc = np.clip(cc, 0, mask.shape[1] - 1)
                    mask[rr, cc] = ann['category_id']

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.long)
            mask = TF.resize(mask.unsqueeze(0), size=image.shape[1:])  # Ensure mask is resized to the same size as the image
            mask = mask.squeeze(0)  # Remove the added channel dimension

        return image, mask
