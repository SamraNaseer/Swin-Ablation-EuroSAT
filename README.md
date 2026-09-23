# Swin-Ablation-EuroSAT
This presents an end-to-end computer vision experimentation pipeline for land-use and land-cover classification using the EuroSAT RGB dataset and the Swin-Tiny Transformer (Swin-T) architecture. 
## Overview
This project implements a controlled baseline-versus-ablation experiment using a pretrained Swin Transformer-Tiny (Swin-T) for EuroSAT RGB satellite image classification.
The modified model investigates two architectural changes:
1. Attention window size: 7×7 → 5×5
2. Activation function: GELU → SiLU
Both models use the same data split and training protocol.
## Dataset
The experiment uses the EuroSAT RGB dataset:
- 27,000 RGB satellite images
- 10 land-use/land-cover classes
- Fixed stratified 80/20 train-validation split
- Split seed: 42
Classes:
- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake
The dataset is not included in this repository because of its size.
Load via torchvision.datasets.EuroSAT(root=’./data’, split=’train’,transform=..., download=True).
## Baseline Model
The baseline uses the pretrained:
`swin_tiny_patch4_window7_224`
The classification head is replaced to produce predictions for 10 EuroSAT classes.
## Architectural Ablation
The modified Swin-T uses:
- 5×5 attention windows instead of 7×7
- SiLU activation instead of GELU
- Pretrained weight transfer
- Resized relative-position bias tables for the changed window size
The two architectural changes are evaluated together as one combined ablation.
<img width="737" height="335" alt="image" src="https://github.com/user-attachments/assets/e6e6c7f9-ea9f-4a00-885a-865d69eb20a8" />
## Training Configuration

| Parameter          | Value            |
|--------------------|------------------|
| Optimizer          | AdamW            |
| Learning Rate      | 5e-5             |
| Weight Decay       | 1e-4             |
| Batch Size         | 32               |
| Epochs             | 30               |
| Warmup             | 5 epochs         |
| Scheduler          | Cosine Annealing |
| Image Size         | 224×224          |
| Gradient Clipping  | 1.0              |
| Split Seed         | 42               |
| Model Seed         | 123              |

### Data Augmentation
Training:
- RandomResizedCrop(224)
- RandomHorizontalFlip
- ImageNet normalization
Validation:
- Resize to 224×224
- ImageNet normalization
Normalization:
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
### Results
### Baseline Swin-T
Best validation accuracy: 99.00%
Best epoch: 26
Final validation accuracy: 98.98%
Maximum recorded global gradient norm: 139.96
### Swin-T 5×5 + SiLU
Best validation accuracy: 98.91%
Best epoch: 25
Final validation accuracy: 98.89%
Maximum recorded global gradient norm: 126.90

The difference in best validation accuracy is approximately 0.09 percentage points.

### Gradient Clipping Analysis

Gradient clipping was applied using a maximum norm of 1.0.

The recorded global gradient norms were above 1.0, particularly during the early training epochs, indicating that gradient clipping was relevant during optimization.

The maximum recorded global gradient norms were:

Baseline: 139.96
Ablation: 126.90

Neither experiment exceeded the 10³ gradient-norm threshold.

Gradient magnitudes generally decreased as training progressed.

### Repository Structure
Swin-Ablation/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── Swin_Ablation.ipynb
│
├── splits/
│   ├── train_indices.npy
│   └── val_indices.npy
│
└── output/
    ├── baseline_swin_t/
    ├── ablation_window5_silu/
    └── comparison_plots/
### Installation
Install the required Python packages:
pip install -r requirements.txt
A CUDA-enabled PyTorch installation is recommended when an NVIDIA GPU is available.
### Running the Experiment
1. Clone this repository.
2. Install the required dependencies.
3. Download and prepare the EuroSAT dataset.
4. Open: Swin_Ablation.ipynb
5. Run the notebook cells in order.
6. The fixed train-validation split is stored in:
- splits/train_indices.npy
- splits/val_indices.npy
7. Experimental results and plots are saved under:
-output/

### Reproducibility
The experiments use fixed random seeds:
Dataset split seed: 42
Model/training seed: 123
The same train-validation split and training hyperparameters are used for both the baseline and ablation experiments.
### Limitations
The 5×5 window and SiLU changes were evaluated together. Therefore, this experiment cannot independently determine the contribution of each modification.
A more complete factorial ablation would separately evaluate:
7×7 + GELU
5×5 + GELU
7×7 + SiLU
5×5 + SiLU
The reported results are from a single controlled training run. Statistical significance and multi-seed variance are not claimed.
### This repository provides:
1. Baseline Swin-T implementation
2. Modified Swin-T architecture
3. Fixed reproducible data split
4. Training configuration
5. Training histories
6. Accuracy and loss plots
7. Gradient norm analysis
8. Baseline-versus-ablation comparison
9. Documentation of architectural changes
10. Experimental findings
