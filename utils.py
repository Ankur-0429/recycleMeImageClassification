import torch
import torchvision
from dataset import SegmentationDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

def save_predictions_as_imgs(
        loader, model, device, folder="saved_images/"
):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.JPG"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/pred_mask_{idx}.JPG")
    model.train()
