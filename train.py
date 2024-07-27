from dataset import SegmentationDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms
from model import UNET
import json

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = SegmentationDataset('./data/annotations_0_train.json', transform)
val_dataset = SegmentationDataset('./data/annotations_0_val.json', transform)
test_dataset = SegmentationDataset('./data/annotations_0_test.json', transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with open('./data/annotations.json') as f:
    data = json.load(f)

model = UNET(in_channels=3, out_channels=len(data['categories'])).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=20):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        print("images length: " + str(len(train_loader)))
        i = 0
        for images, masks in train_loader:
            print(i)
            i+=1
            images = images.to(device)
            masks = masks.to(device).squeeze(1)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

train(model, train_loader, val_loader, criterion, optimizer, device, epochs=1)

torch.save(model.state_dict(), "unet_model.pth")
