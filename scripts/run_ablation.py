# ============================================================
# REPRODUCIBLE SWIN-T ABLATION EXPERIMENT
# Changes: 7x7 -> 5x5 window, GELU -> SiLU
# Extracted from the validated Swin_Ablation.ipynb
# ============================================================

import torch
import torch.nn as nn
import timm

from timm.models.swin_transformer import SwinTransformer

# ============================================================
# MODIFIED SWIN-T
# Changes:
#   1. Window size: 7x7 -> 5x5
#   2. Activation: GELU -> SiLU
# ============================================================

NUM_CLASSES = 10

modified_model = SwinTransformer(
    img_size=224,
    patch_size=4,
    in_chans=3,
    num_classes=NUM_CLASSES,
    global_pool='avg',
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    window_size=5,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    proj_drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
)

# ------------------------------------------------------------
# Change GELU -> SiLU in every MLP block
# ------------------------------------------------------------

silu_count = 0

for module in modified_model.modules():
    if hasattr(module, "act") and isinstance(module.act, nn.GELU):
        module.act = nn.SiLU()
        silu_count += 1

print("=" * 60)
print("MODIFIED SWIN-T CREATED")
print("=" * 60)

print("Model type:", type(modified_model).__name__)
print("Window size: 5x5")
print("Activation: SiLU")
print("Number of SiLU activations:", silu_count)
print("Number of classes:", NUM_CLASSES)

print("=" * 60)

# Check window sizes and activation functions

print("\nChecking Swin blocks...")

for name, module in modified_model.named_modules():
    if hasattr(module, "window_size") and hasattr(module, "mlp"):
        print(
            name,
            "| Window:", module.window_size,
            "| Activation:", module.mlp.act.__class__.__name__
        )

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ============================================================
# RECREATE DATA LOADERS USING THE SAVED SPLIT
# ============================================================

DATA_DIR = "./data"
SPLIT_DIR = "./splits"

BATCH_SIZE = 32
NUM_WORKERS = 0

# Same normalization required by the assignment
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Training transform
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Validation transform
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Load EuroSAT twice so train/validation can have different transforms
train_dataset_full = datasets.EuroSAT(
    root=DATA_DIR,
    download=False,
    transform=train_transform
)

val_dataset_full = datasets.EuroSAT(
    root=DATA_DIR,
    download=False,
    transform=val_transform
)

# Load the SAME fixed split used for the baseline
train_indices = np.load(
    os.path.join(SPLIT_DIR, "train_indices.npy")
)

val_indices = np.load(
    os.path.join(SPLIT_DIR, "val_indices.npy")
)

train_dataset = Subset(train_dataset_full, train_indices)
val_dataset = Subset(val_dataset_full, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("=" * 60)
print("DATA LOADERS READY")
print("=" * 60)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Batch size:", BATCH_SIZE)
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

modified_model = modified_model.to(device)
modified_model.train()

images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)

print("Input shape:", images.shape)
print("Labels shape:", labels.shape)

outputs = modified_model(images)

print("Output shape:", outputs.shape)

criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)

print("Loss:", loss.item())

modified_model.zero_grad(set_to_none=True)
loss.backward()

print("Backward pass: SUCCESS")

print("=" * 60)
print("MODIFIED MODEL SANITY TEST PASSED")
print("=" * 60)

# ============================================================
# LOAD ORIGINAL PRETRAINED SWIN-T
# This model is only used as the source of pretrained weights.
# ============================================================

import timm
import torch

print("Loading original pretrained Swin-T...")

pretrained_model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=10
)

print("Pretrained Swin-T loaded successfully.")
print("Source model:", type(pretrained_model).__name__)

# ============================================================
# TRANSFER PRETRAINED WEIGHTS
# 7x7 Swin-T -> 5x5 Swin-T
#
# Relative-position bias tables are resized from:
#       13 x 13  ->  9 x 9
# because:
#       2*7 - 1 = 13
#       2*5 - 1 = 9
# ============================================================

import torch
import torch.nn.functional as F

source_state = pretrained_model.state_dict()
target_state = modified_model.state_dict()

new_state = {}

transferred = 0
resized = 0
skipped = []

for key, target_tensor in target_state.items():

    if key not in source_state:
        skipped.append(key)
        continue

    source_tensor = source_state[key]

    # --------------------------------------------------------
    # Relative position bias table:
    # pretrained: [169, num_heads]  -> 13x13
    # modified:   [81, num_heads]   ->  9x9
    # --------------------------------------------------------
    if "relative_position_bias_table" in key:

        num_heads = target_tensor.shape[1]

        source_h = int(source_tensor.shape[0] ** 0.5)
        target_h = int(target_tensor.shape[0] ** 0.5)

        if (
            source_h * source_h == source_tensor.shape[0]
            and target_h * target_h == target_tensor.shape[0]
        ):
            # [169, heads] -> [heads, 13, 13]
            bias = source_tensor.transpose(0, 1).reshape(
                1, num_heads, source_h, source_h
            )

            # Resize 13x13 -> 9x9
            bias = F.interpolate(
                bias,
                size=(target_h, target_h),
                mode="bicubic",
                align_corners=False
            )

            # [1, heads, 9, 9] -> [81, heads]
            bias = bias.reshape(
                num_heads, target_h * target_h
            ).transpose(0, 1)

            new_state[key] = bias
            resized += 1
            continue

    # --------------------------------------------------------
    # Normal parameters with matching shape
    # --------------------------------------------------------
    if source_tensor.shape == target_tensor.shape:
        new_state[key] = source_tensor
        transferred += 1
    else:
        skipped.append(
            f"{key} | source {tuple(source_tensor.shape)} "
            f"-> target {tuple(target_tensor.shape)}"
        )

# Load transferred weights
missing, unexpected = modified_model.load_state_dict(
    new_state,
    strict=False
)

print("=" * 65)
print("PRETRAINED WEIGHT TRANSFER COMPLETE")
print("=" * 65)

print("Normal parameters transferred:", transferred)
print("Relative-position tables resized:", resized)
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))
print("Skipped/mismatched parameters:", len(skipped))

if skipped:
    print("\nFirst few skipped/mismatched parameters:")
    for item in skipped[:10]:
        print(" -", item)

print("=" * 65)

# ============================================================
# FINAL PRE-TRAINING SANITY CHECK
# ============================================================

modified_model.eval()

with torch.no_grad():
    test_images, test_labels = next(iter(val_loader))

    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    test_outputs = modified_model(test_images)

print("=" * 60)
print("FINAL MODEL CHECK")
print("=" * 60)

print("Input shape :", test_images.shape)
print("Output shape:", test_outputs.shape)
print("Output contains NaN:", torch.isnan(test_outputs).any().item())
print("Output contains Inf:", torch.isinf(test_outputs).any().item())

print("=" * 60)
print("READY FOR TRAINING")
print("=" * 60)

# ============================================================
# ABLATION EXPERIMENT
# Swin-T: 7x7 GELU -> 5x5 SiLU
# ============================================================

import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 123

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# 2. EXPERIMENT SETTINGS
# ============================================================

EXPERIMENT_NAME = "ablation_window5_silu"

OUTPUT_DIR = os.path.join(
    "./outputs",
    EXPERIMENT_NAME
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 10
EPOCHS = 30
BATCH_SIZE = 32

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

WARMUP_EPOCHS = 5
GRAD_CLIP = 1.0

# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("STARTING ABLATION EXPERIMENT")
print("=" * 70)

print("Experiment:", EXPERIMENT_NAME)
print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Warmup epochs:", WARMUP_EPOCHS)
print("Gradient clipping:", GRAD_CLIP)
print("Architecture: Swin-T + 5x5 windows + SiLU")
print("=" * 70)

# ============================================================
# 4. MOVE MODEL TO GPU
# ============================================================

modified_model = modified_model.to(device)

# ============================================================
# 5. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    modified_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# ============================================================
# 6. LEARNING-RATE SCHEDULE
# 5-EPOCH LINEAR WARMUP + COSINE DECAY
# ============================================================

def get_learning_rate(epoch):

    # epoch is 1-based

    if epoch <= WARMUP_EPOCHS:

        # Start at 20% of LR and linearly increase
        warmup_start = LEARNING_RATE * 0.2

        progress = (
            (epoch - 1)
            / max(1, WARMUP_EPOCHS - 1)
        )

        return (
            warmup_start
            + progress * (LEARNING_RATE - warmup_start)
        )

    else:

        progress = (
            (epoch - WARMUP_EPOCHS)
            / max(1, EPOCHS - WARMUP_EPOCHS)
        )

        return (
            LEARNING_RATE
            * 0.5
            * (1 + np.cos(np.pi * progress))
        )


def set_learning_rate(optimizer, lr):

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ============================================================
# 7. GRADIENT STATISTICS
# ============================================================

def calculate_gradient_statistics(model):

    grad_norms = []

    for parameter in model.parameters():

        if parameter.grad is not None:

            grad_norms.append(
                parameter.grad.detach().norm(2).item()
            )

    if len(grad_norms) == 0:

        return 0.0, 0.0, 0.0

    avg_grad_norm = float(np.mean(grad_norms))
    max_grad_norm = float(np.max(grad_norms))

    global_grad_norm = float(
        np.sqrt(
            sum(x ** 2 for x in grad_norms)
        )
    )

    return (
        avg_grad_norm,
        global_grad_norm,
        max_grad_norm
    )


# ============================================================
# 8. TRAINING HISTORY
# ============================================================

history = []

best_val_accuracy = -1.0
best_epoch = -1

# ============================================================
# 9. TRAINING LOOP
# ============================================================

for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    current_lr = get_learning_rate(epoch)

    set_learning_rate(
        optimizer,
        current_lr
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    modified_model.train()

    train_loss_total = 0.0
    train_correct = 0
    train_total = 0

    epoch_avg_gradients = []
    epoch_global_gradients = []
    epoch_max_gradients = []
    epoch_max_global_gradient = 0.0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # Forward
        outputs = modified_model(images)

        loss = criterion(
            outputs,
            labels
        )

        # Backward
        loss.backward()

        # Gradient statistics BEFORE clipping
        (
            avg_grad,
            global_grad,
            max_grad
        ) = calculate_gradient_statistics(
            modified_model
        )

        epoch_avg_gradients.append(
            avg_grad
        )

        epoch_global_gradients.append(
            global_grad
        )

        epoch_max_gradients.append(
            max_grad
        )

        epoch_max_global_gradient = max(
            epoch_max_global_gradient,
            global_grad
        )

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            modified_model.parameters(),
            GRAD_CLIP
        )

        # Optimizer
        optimizer.step()

        # Statistics
        train_loss_total += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss = (
        train_loss_total
        / train_total
    )

    train_accuracy = (
        100.0
        * train_correct
        / train_total
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    modified_model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = modified_model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = (
        val_loss_total
        / val_total
    )

    val_accuracy = (
        100.0
        * val_correct
        / val_total
    )

    # --------------------------------------------------------
    # GRADIENT SUMMARY
    # --------------------------------------------------------

    avg_grad_norm = float(
        np.mean(epoch_avg_gradients)
    )

    avg_global_grad_norm = float(
        np.mean(epoch_global_gradients)
    )

    avg_max_grad_norm = float(
        np.mean(epoch_max_gradients)
    )

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    epoch_time = (
        time.time()
        - epoch_start
    )

    # --------------------------------------------------------
    # SAVE HISTORY ROW
    # --------------------------------------------------------

    row = {
        "epoch": epoch,
        "learning_rate": current_lr,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "avg_grad_norm": avg_grad_norm,
        "avg_global_grad_norm": avg_global_grad_norm,
        "avg_max_grad_norm": avg_max_grad_norm,
        "max_global_grad_norm": epoch_max_global_gradient,
        "epoch_time_seconds": epoch_time
    }

    history.append(row)

    # --------------------------------------------------------
    # SAVE HISTORY
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
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch

        torch.save(
            modified_model.state_dict(),
            os.path.join(
                OUTPUT_DIR,
                "best_model.pth"
            )
        )

        best_marker = " <-- BEST"

    else:

        best_marker = ""

    # --------------------------------------------------------
    # SAVE LATEST CHECKPOINT
    # --------------------------------------------------------

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": modified_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "history": history
        },
        os.path.join(
            OUTPUT_DIR,
            "latest_checkpoint.pth"
        )
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nEpoch {epoch:02d}/{EPOCHS} "
        f"| LR={current_lr:.8f}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Acc: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Acc: {val_accuracy:.2f}%"
        f"{best_marker}"
    )

    print(
        f"Avg Grad Norm: {avg_grad_norm:.4f}"
    )

    print(
        f"Global Grad Norm: {avg_global_grad_norm:.4f}"
    )

    print(
        f"Max Grad Norm: {avg_max_grad_norm:.4f}"
    )

    print(
        f"Epoch Time: {epoch_time / 60:.2f} min"
    )


# ============================================================
# 10. SAVE CONFIGURATION
# ============================================================

config = {
    "experiment_name": EXPERIMENT_NAME,
    "architecture": "Swin-T",
    "window_size": 5,
    "activation": "SiLU",
    "num_classes": NUM_CLASSES,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "warmup_epochs": WARMUP_EPOCHS,
    "gradient_clipping": GRAD_CLIP,
    "image_size": 224,
    "optimizer": "AdamW",
    "split_seed": 42,
    "model_seed": SEED,
    "train_images": len(train_dataset),
    "validation_images": len(val_dataset)
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "config.json"
    ),
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("ABLATION TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best Epoch: {best_epoch}"
)

print(
    "History:",
    os.path.join(
        OUTPUT_DIR,
        "history.csv"
    )
)

print(
    "Best Model:",
    os.path.join(
        OUTPUT_DIR,
        "best_model.pth"
    )
)

print("=" * 70)