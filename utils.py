import torch
import numpy as np
from dataset import SegmentationDataset
from torch.utils.data import DataLoader
import os
from PIL import Image

def save_checkpoint(state, filename="checkpoint.ptch.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    annotations_train_file,
    annotations_val_file,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = SegmentationDataset(annotations_file=annotations_train_file, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    val_ds = SegmentationDataset(annotations_file=annotations_val_file, transform=val_transform)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader

# There are better metrics
# perhaps using dice score?
# This accuracy only works for binary operation
def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score = (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(
        f"Dice score: {dice_score/len(loader)}"
    )
    model.train()

def save_predictions_as_imgs(loader, model, device, folder="saved_images/"):
    model.eval()
    os.makedirs(folder, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = torch.argmax(preds, dim=1)

        # Save input images
        for i in range(x.shape[0]):
            input_img = x[i].cpu().numpy().transpose(1, 2, 0)  # Convert from [C, H, W] to [H, W, C]
            input_img = (input_img * 255).astype(np.uint8)  # Convert to uint8
            input_img = Image.fromarray(input_img)
            input_img.save(f"{folder}/input_{idx}_{i}.png")

        # Save predicted masks
        for i in range(preds.shape[0]):
            pred_img = preds[i].cpu().numpy().astype(np.uint8)
            pred_img = Image.fromarray(pred_img)
            pred_img.save(f"{folder}/pred_{idx}_{i}.png")

        # Save ground truth masks
        for i in range(y.shape[0]):
            mask_img = y[i].cpu().numpy().astype(np.uint8)
            mask_img = Image.fromarray(mask_img)
            mask_img.save(f"{folder}/mask_{idx}_{i}.png")

    model.train()