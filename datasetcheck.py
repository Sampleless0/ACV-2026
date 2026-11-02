
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASET_PATH = Path(
    r"C:\Users\euris\Documents\Projects\ACV-2026\raw_images\PlantVillage\color"
)

# 1. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 2. Load all images and their folder-based labels
dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

# 3. Create batches of 32 images
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# 4. Retrieve one batch
images, labels = next(iter(loader))

print("Total images:", len(dataset))
print("Number of classes:", len(dataset.classes))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("\nFirst 5 class names:")
print(dataset.classes[:5])

print("\nFirst 5 labels:")
print(labels[:5])
