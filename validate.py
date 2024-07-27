import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import UNET  # Import your UNET model
from dataset import SegmentationDataset  # Import your dataset class
import torch.nn as nn
import json

# Set device to MPS if available, otherwise use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define the same transformations used during training
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create DataLoader for validation or test dataset
test_dataset = SegmentationDataset('./data/annotations_0_test.json', transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

with open('./data/annotations.json') as f:
    data = json.load(f)

# Load the saved model state
model = UNET(in_channels=3, out_channels=len(data['categories'])).to(device)
model.load_state_dict(torch.load("unet_model.pth", map_location=device), strict=True)
model.eval()  # Set the model to evaluation mode

criterion = nn.CrossEntropyLoss()

def validate(model, test_loader, criterion, device):
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss/len(test_loader)}")

# Validate the model
validate(model, test_loader, criterion, device)
