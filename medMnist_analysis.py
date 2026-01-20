import os
import torch
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import Counter
import torch
import math

def compute_imbalance_metrics(labels, num_classes):
    """
    labels: 1D torch tensor of shape [N]
    num_classes: int
    """

    counts = torch.bincount(labels, minlength=num_classes).float()
    
    max_c = counts.max()
    min_c = counts[counts > 0].min()  # avoid zero-class issue
    mean_c = counts.mean()
    std_c = counts.std()

    # 1. Imbalance Ratio
    imbalance_ratio = (max_c / min_c).item()

    # 2. Max-Min Ratio
    max_min_ratio = (max_c - min_c).item()

    # 3. Coefficient of Variation
    cv = (std_c / mean_c).item()

    # 4. Normalized Entropy
    probs = counts / counts.sum()
    entropy = -(probs * torch.log(probs + 1e-8)).sum()
    max_entropy = math.log(num_classes)
    normalized_entropy = (entropy / max_entropy).item()

    # 5. Gini Index
    sorted_counts = torch.sort(counts)[0]
    n = num_classes
    index = torch.arange(1, n + 1).float()
    gini = ((2 * index - n - 1) * sorted_counts).sum() / (n * sorted_counts.sum())
    gini = gini.item()

    return {
        "Class Counts": counts.tolist(),
        "Imbalance Ratio": imbalance_ratio,
        "Max-Min Difference": max_min_ratio,
        "Coefficient of Variation": cv,
        "Normalized Entropy": normalized_entropy,
        "Gini Index": gini
    }



# -------------------------
# Directory oct to save dataset
# -------------------------
dir_oct = "/home/gen/yash/OurData/data/"   # <-- CHANGE THIS
os.makedirs(dir_oct, exist_ok=True)

# -------------------------
# Load octmnist (train + val + test)
# -------------------------
info = INFO["octmnist"]
DataClass = getattr(medmnist, info["python_class"])

transform = transforms.Compose([transforms.Resize(size=(64,64)),transforms.ToTensor()])

splits = ["train", "val", "test"]

datasets = [
    DataClass(split=s, transform=transform, download=True, root=dir_oct)
    for s in splits
]

# -------------------------
# DataLoader
# -------------------------
loader = DataLoader(
    torch.utils.data.ConcatDataset(datasets),
    batch_size=256,
    shuffle=False,
    num_workers=2
)

# -------------------------
# Compute mean and std
# -------------------------
mean = 0.0
std = 0.0
total_pixels = 0

print("Computing mean & std ...")

for imgs, _ in tqdm(loader):
    # imgs shape: [B, 1, 28, 28]
    print(imgs.shape)
    pixels = imgs.numel() / imgs.shape[1]  # total pixels per channel
    mean += imgs.mean([0, 2, 3]) * pixels
    std += imgs.std([0, 2, 3]) * pixels
    total_pixels += pixels

mean /= total_pixels
std /= total_pixels
print("\n=== octmnist Mean & Std ===")
print("Mean : ")
print(mean)
print("Std  : ")
print(std)

print("\nData stored in:", dir_oct)


def count_images_per_class(split="train", root="/home/gen/yash/OurData/data/"):
    info = INFO["octmnist"]
    DataClass = getattr(medmnist, info["python_class"])

    dataset = DataClass(split=split, download=True, root=root)

    counter = Counter()
    for _, target in dataset:
        counter[int(target)] += 1

    return dict(counter)

train_counts = count_images_per_class(split="train")
val_counts = count_images_per_class(split="val")
test_counts = count_images_per_class(split="test")

print("Train:", train_counts)
print("Val:", val_counts)
print("Test:", test_counts)

info = INFO["octmnist"]
DataClass = getattr(medmnist, info["python_class"])

dataset = DataClass(split="train", download=False, root="/home/gen/yash/OurData/data/")
labels = torch.tensor(dataset.labels).squeeze()
num_classes = torch.unique(labels).numel()

metrics = compute_imbalance_metrics(labels, num_classes=num_classes)
for k, v in metrics.items():
    print(f"{k}: {v}")
