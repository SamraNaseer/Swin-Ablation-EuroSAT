# ============================================================
# BASELINE VS ABLATION COMPARISON PLOTS
# Extracted from the validated Swin_Ablation.ipynb
# ============================================================

# ============================================================
# BASELINE vs ABLATION COMPARISON PLOTS
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASELINE_DIR = "./outputs/baseline_swin_t"
ABLATION_DIR = "./outputs/ablation_window5_silu"

PLOT_DIR = "./outputs/comparison_plots"

os.makedirs(PLOT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Load histories
# ------------------------------------------------------------

baseline = pd.read_csv(
    os.path.join(BASELINE_DIR, "history.csv")
)

ablation = pd.read_csv(
    os.path.join(ABLATION_DIR, "history.csv")
)

print("Baseline rows:", len(baseline))
print("Ablation rows:", len(ablation))

print("\nBaseline columns:")
print(baseline.columns.tolist())

print("\nAblation columns:")
print(ablation.columns.tolist())


# ============================================================
# 1. OVERLAID TRAINING LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    baseline["epoch"],
    baseline["train_loss"],
    label="Baseline Swin-T"
)

plt.plot(
    ablation["epoch"],
    ablation["train_loss"],
    label="5x5 + SiLU"
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss: Baseline vs Ablation")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "comparison_train_loss.png"
    ),
    dpi=300
)

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "comparison_train_loss.svg"
    )
)

plt.show()


# ============================================================
# 2. OVERLAID VALIDATION LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    baseline["epoch"],
    baseline["val_loss"],
    label="Baseline Swin-T"
)

plt.plot(
    ablation["epoch"],
    ablation["val_loss"],
    label="5x5 + SiLU"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss: Baseline vs Ablation")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "comparison_val_loss.png"
    ),
    dpi=300
)

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "comparison_val_loss.svg"
    )
)

plt.show()


# ============================================================
# 3. OVERLAID VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    baseline["epoch"],
    baseline["val_accuracy"],
    label="Baseline Swin-T"
)

plt.plot(
    ablation["epoch"],
    ablation["val_accuracy"],
    label="5x5 + SiLU"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy: Baseline vs Ablation")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "comparison_val_accuracy.png"
    ),
    dpi=300
)

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "comparison_val_accuracy.svg"
    )
)

plt.show()


# ============================================================
# 4. OVERLAID AVERAGE GLOBAL GRADIENT NORM
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    baseline["epoch"],
    baseline["avg_global_grad_norm"],
    label="Baseline Swin-T"
)

plt.plot(
    ablation["epoch"],
    ablation["avg_global_grad_norm"],
    label="5x5 + SiLU"
)

plt.xlabel("Epoch")
plt.ylabel("Average Global Gradient Norm")
plt.title("Gradient Norm: Baseline vs Ablation")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "comparison_gradient_norm.png"
    ),
    dpi=300
)

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "comparison_gradient_norm.svg"
    )
)

plt.show()


# ============================================================
# 5. MAXIMUM GLOBAL GRADIENT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    baseline["epoch"],
    baseline["max_global_grad_norm"],
    label="Baseline Swin-T"
)

plt.plot(
    ablation["epoch"],
    ablation["max_global_grad_norm"],
    label="5x5 + SiLU"
)

plt.axhline(
    y=1000,
    linestyle="--",
    label="10³ threshold"
)

plt.xlabel("Epoch")
plt.ylabel("Maximum Global Gradient Norm")
plt.title("Maximum Global Gradient Norm")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "comparison_max_gradient.png"
    ),
    dpi=300
)

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "comparison_max_gradient.svg"
    )
)

plt.show()


# ============================================================
# 6. CREATE NUMERICAL SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Model": [
        "Baseline Swin-T",
        "Swin-T 5x5 + SiLU"
    ],

    "Best Val Accuracy (%)": [
        baseline["val_accuracy"].max(),
        ablation["val_accuracy"].max()
    ],

    "Best Epoch": [
        baseline.loc[
            baseline["val_accuracy"].idxmax(),
            "epoch"
        ],
        ablation.loc[
            ablation["val_accuracy"].idxmax(),
            "epoch"
        ]
    ],

    "Best Val Loss": [
        baseline["val_loss"].min(),
        ablation["val_loss"].min()
    ],

    "Final Val Accuracy (%)": [
        baseline.iloc[-1]["val_accuracy"],
        ablation.iloc[-1]["val_accuracy"]
    ],

    "Maximum Global Gradient": [
        baseline["max_global_grad_norm"].max(),
        ablation["max_global_grad_norm"].max()
    ]
})

print("\n")
print("=" * 80)
print("BASELINE vs ABLATION SUMMARY")
print("=" * 80)

display(summary)

summary.to_csv(
    os.path.join(
        PLOT_DIR,
        "baseline_vs_ablation_summary.csv"
    ),
    index=False
)

print("\nSaved comparison files to:")
print(PLOT_DIR)