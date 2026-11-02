
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch import nn

# Step 1: Select GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

# Step 2: Load one leaf image
DATASET_PATH = Path(
    r"C:\Users\euris\Documents\Projects\ACV-2026\raw_images\PlantVillage\color"
)

image_path = next(
    p for p in DATASET_PATH.rglob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))
pixels = np.array(image)

# Step 3: Convert image into tensor
x = torch.from_numpy(pixels)
x = x.permute(2, 0, 1)
x = x.unsqueeze(0).float() / 255.0
x = x.to(device)

# Step 4: Build a small CNN
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(32, 38)
).to(device)

# Step 5: Forward propagation
model.eval()

print("Input shape:", x.shape)

with torch.inference_mode():
    for layer in model:
        x = layer(x)
        print(layer.__class__.__name__, x.shape)
