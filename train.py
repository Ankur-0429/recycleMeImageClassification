from dataset import SegmentationDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms
from model import UNET
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from tqdm import tqdm
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)

with open('./data/annotations.json') as f:
    CATEGORIES = json.load(f)["categories"]


LEARNING_RATE=1e-4
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
BATCH_SIZE=32
NUM_EPOCHS=1
NUM_WORKERS=2
IMAGE_HEIGHT=160
IMAGE_WIDTH=240
PIN_MEMORY=True
LOAD_MODEL=False

def train(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        predictions = model(data)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    model = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        annotations_train_file="annotations_0_test.json",
        annotations_val_file="annotations_0_val.json",
        batch_size=BATCH_SIZE,
        train_transform=train_transform,
        val_transform=val_transforms,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    for _ in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)
        save_predictions_as_imgs(
            val_loader, model, device=DEVICE
        )

if __name__ == "__main__":
    main()


