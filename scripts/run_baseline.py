# ============================================================
# REPRODUCIBLE BASELINE SWIN-T EXPERIMENT
# Extracted from the validated Swin_Ablation.ipynb
# ============================================================

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
BATCH_SIZE = 32
IMAGE_SIZE = 224

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# 2. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# 3. DOWNLOAD EURO-SAT
# ============================================================

data_dir = "./data"

print("Downloading/loading EuroSAT...")

dataset_for_split = datasets.EuroSAT(
    root=data_dir,
    download=True,
    transform=None
)

print("Dataset loaded!")
print("Total images:", len(dataset_for_split))
print("Number of classes:", len(dataset_for_split.classes))
print("Classes:")
print(dataset_for_split.classes)

# ============================================================
# 4. CREATE FIXED STRATIFIED 80/20 TRAIN-VALIDATION SPLIT
# ============================================================

from sklearn.model_selection import train_test_split
import numpy as np
import os

SEED = 42

# Get labels for every image
labels = np.array(dataset_for_split.targets)

# Create indices
all_indices = np.arange(len(dataset_for_split))

# 80% train / 20% validation
train_indices, val_indices = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)

print("Split created successfully!")
print("Total samples:", len(all_indices))
print("Training samples:", len(train_indices))
print("Validation samples:", len(val_indices))

# Check percentages
print(f"\nTraining percentage: {len(train_indices)/len(all_indices)*100:.2f}%")
print(f"Validation percentage: {len(val_indices)/len(all_indices)*100:.2f}%")

# ============================================================
# 5. CHECK CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")
print("-" * 60)

for class_id, class_name in enumerate(dataset_for_split.classes):

    total_count = np.sum(labels == class_id)
    train_count = np.sum(labels[train_indices] == class_id)
    val_count = np.sum(labels[val_indices] == class_id)

    print(
        f"{class_name:25s} | "
        f"Total: {total_count:4d} | "
        f"Train: {train_count:4d} | "
        f"Val: {val_count:4d}"
    )

# ============================================================
# 6. SAVE SPLIT INDICES
# ============================================================

os.makedirs("./splits", exist_ok=True)

np.save("./splits/train_indices.npy", train_indices)
np.save("./splits/val_indices.npy", val_indices)

print("\nSplit indices saved!")
print("./splits/train_indices.npy")
print("./splits/val_indices.npy")

# ============================================================
# 6. CREATE TRAINING AND VALIDATION DATASETS
# ============================================================

from torch.utils.data import DataLoader, Subset

# ------------------------------------------------------------
# Training dataset
# ------------------------------------------------------------

train_dataset_full = datasets.EuroSAT(
    root="./data",
    download=False,
    transform=train_transform
)

# ------------------------------------------------------------
# Validation dataset
# ------------------------------------------------------------

val_dataset_full = datasets.EuroSAT(
    root="./data",
    download=False,
    transform=val_transform
)

# Use the SAME indices we created earlier
train_dataset = Subset(
    train_dataset_full,
    train_indices
)

val_dataset = Subset(
    val_dataset_full,
    val_indices
)

# ============================================================
# 7. CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ============================================================
# 8. CHECK
# ============================================================

print("DataLoaders created successfully!")
print()
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Batch size:", 32)
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

# ============================================================
# BASELINE SWIN-T TRAINING
# Advanced Computer Vision Assignment
# ============================================================

import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 123

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# 2. CONFIGURATION
# ============================================================

NUM_CLASSES = 10
EPOCHS = 30
BATCH_SIZE = 32

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

WARMUP_EPOCHS = 5
GRAD_CLIP = 1.0

OUTPUT_DIR = "./outputs/baseline_swin_t"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("BASELINE SWIN-T EXPERIMENT")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Warmup epochs:", WARMUP_EPOCHS)
print("Gradient clipping:", GRAD_CLIP)

# ============================================================
# 3. CREATE PRETRAINED SWIN-T
# ============================================================

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=NUM_CLASSES
)

model = model.to(device)

print("\nModel:")
print("Swin-Tiny Patch4 Window7 224")
print("Number of classes:", NUM_CLASSES)

# ============================================================
# 4. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()

# ============================================================
# 5. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# ============================================================
# 6. LEARNING-RATE SCHEDULER
#
# 5-epoch linear warmup followed by cosine decay.
# ============================================================

def get_learning_rate(epoch):
    """
    Returns learning rate for the current epoch.

    Epoch numbering starts at 1.
    """

    # -------------------------
    # Warmup
    # -------------------------
    if epoch <= WARMUP_EPOCHS:

        # Start at 20% of target LR
        # and linearly increase to target LR.
        warmup_start = LEARNING_RATE * 0.2

        alpha = epoch / WARMUP_EPOCHS

        lr = warmup_start + alpha * (
            LEARNING_RATE - warmup_start
        )

    # -------------------------
    # Cosine decay
    # -------------------------
    else:

        progress = (
            (epoch - WARMUP_EPOCHS)
            / (EPOCHS - WARMUP_EPOCHS)
        )

        lr = LEARNING_RATE * 0.5 * (
            1 + np.cos(np.pi * progress)
        )

    return lr


def set_learning_rate(optimizer, lr):

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ============================================================
# 7. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    grad_clip
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # Gradient statistics
    grad_norms_before_clip = []
    global_grad_norms = []
    max_grad_norms = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # Clear gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # ----------------------------------------------------
        # Calculate gradient norms BEFORE clipping
        # ----------------------------------------------------

        parameter_grad_norms = []

        for parameter in model.parameters():

            if parameter.grad is not None:

                norm = parameter.grad.detach().norm(2).item()

                parameter_grad_norms.append(norm)

        if parameter_grad_norms:

            parameter_grad_norms = np.array(
                parameter_grad_norms
            )

            avg_grad = parameter_grad_norms.mean()

            max_grad = parameter_grad_norms.max()

            global_grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip
            )

            global_grad = float(global_grad)

            grad_norms_before_clip.append(avg_grad)
            global_grad_norms.append(global_grad)
            max_grad_norms.append(max_grad)

        else:

            grad_norms_before_clip.append(0.0)
            global_grad_norms.append(0.0)
            max_grad_norms.append(0.0)

        # ----------------------------------------------------
        # Optimizer update
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    avg_grad_norm = float(
        np.mean(grad_norms_before_clip)
    )

    avg_global_grad_norm = float(
        np.mean(global_grad_norms)
    )

    avg_max_grad_norm = float(
        np.mean(max_grad_norms)
    )

    maximum_global_grad_norm = float(
        np.max(global_grad_norms)
    )

    return {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
        "avg_grad_norm": avg_grad_norm,
        "avg_global_grad_norm": avg_global_grad_norm,
        "avg_max_grad_norm": avg_max_grad_norm,
        "max_global_grad_norm": maximum_global_grad_norm
    }


# ============================================================
# 8. VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    loss = running_loss / total

    accuracy = 100.0 * correct / total

    return loss, accuracy


# ============================================================
# 9. SAVE EXPERIMENT CONFIGURATION
# ============================================================

config = {
    "experiment": "baseline_swin_t",
    "dataset": "EuroSAT RGB",
    "num_classes": NUM_CLASSES,
    "train_samples": len(train_dataset),
    "validation_samples": len(val_dataset),
    "split": "80/20 stratified",
    "split_seed": 42,
    "model_seed": SEED,
    "model": "swin_tiny_patch4_window7_224",
    "pretrained": True,
    "image_size": 224,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "warmup_epochs": WARMUP_EPOCHS,
    "scheduler": "5-epoch warmup + cosine decay",
    "gradient_clipping": GRAD_CLIP
}

with open(
    os.path.join(OUTPUT_DIR, "config.json"),
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )

print("\nConfiguration saved.")


# ============================================================
# 10. TRAINING LOOP
# ============================================================

history = []

best_val_accuracy = -1.0
best_epoch = -1

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    start_time = time.time()

    # Set LR
    current_lr = get_learning_rate(epoch)

    set_learning_rate(
        optimizer,
        current_lr
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_results = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        grad_clip=GRAD_CLIP
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loss, val_accuracy = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device
    )

    epoch_time = time.time() - start_time

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    row = {
        "epoch": epoch,
        "learning_rate": current_lr,

        "train_loss": train_results["loss"],
        "train_accuracy": train_results["accuracy"],

        "val_loss": val_loss,
        "val_accuracy": val_accuracy,

        "avg_grad_norm": train_results["avg_grad_norm"],
        "avg_global_grad_norm": train_results["avg_global_grad_norm"],
        "avg_max_grad_norm": train_results["avg_max_grad_norm"],
        "max_global_grad_norm": train_results["max_global_grad_norm"],

        "epoch_time_seconds": epoch_time
    }

    history.append(row)

    # --------------------------------------------------------
    # Save CSV after EVERY epoch
    # --------------------------------------------------------

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "history.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Save latest checkpoint
    # --------------------------------------------------------

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "history": history
        },
        os.path.join(
            OUTPUT_DIR,
            "latest_checkpoint.pth"
        )
    )

    # --------------------------------------------------------
    # Save BEST checkpoint
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "history": history
            },
            os.path.join(
                OUTPUT_DIR,
                "best_model.pth"
            )
        )

        best_marker = " <-- BEST"

    else:

        best_marker = ""

    # --------------------------------------------------------
    # Print epoch
    # --------------------------------------------------------

    print(
        f"\nEpoch {epoch:02d}/{EPOCHS} | "
        f"LR={current_lr:.8f}"
    )

    print(
        f"Train Loss: {train_results['loss']:.4f} | "
        f"Train Acc: {train_results['accuracy']:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
        f"{best_marker}"
    )

    print(
        f"Avg Grad Norm: "
        f"{train_results['avg_grad_norm']:.4f}"
    )

    print(
        f"Global Grad Norm: "
        f"{train_results['avg_global_grad_norm']:.4f}"
    )

    print(
        f"Max Grad Norm: "
        f"{train_results['avg_max_grad_norm']:.4f}"
    )

    print(
        f"Epoch Time: "
        f"{epoch_time / 60:.2f} min"
    )

# ============================================================
# 11. FINAL SUMMARY
# ============================================================

history_df = pd.DataFrame(history)

print("\n" + "=" * 70)
print("BASELINE TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best Epoch: "
    f"{best_epoch}"
)

print(
    f"History saved to: "
    f"{OUTPUT_DIR}/history.csv"
)

print(
    f"Best model saved to: "
    f"{OUTPUT_DIR}/best_model.pth"
)
