import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from model import UNET  # Import your UNET model
import matplotlib.pyplot as plt
import json

def segment_image(image_path, model, device):
    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Run inference
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(image)
    
    # Convert output tensor to numpy array
    output = output.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
    output = np.argmax(output, axis=0)  # Assuming the output is in the form of class probabilities

    # Convert to PIL Image for visualization
    output_image = Image.fromarray(output.astype(np.uint8))

    # Display the image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(output_image, cmap='gray')  # Use 'gray' colormap for binary or segmented images
    plt.axis('off')  # Hide axis
    plt.show()

    return output_image

# Example usage in a Jupyter Notebook cell
if __name__ == "__main__":
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    with open('./data/annotations.json') as f:
        data = json.load(f)
    model = UNET(in_channels=3, out_channels=len(data["categories"])).to(device)  # Update this as needed
    model.load_state_dict(torch.load("unet_model.pth", map_location=device))

    # Path to the image you want to segment
    image_path = './data/batch_5/000000.JPG'
    
    segmented_image = segment_image(image_path, model, device)
