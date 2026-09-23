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
## Training Configuration
<img width="737" height="335" alt="image" src="https://github.com/user-attachments/assets/e6e6c7f9-ea9f-4a00-885a-865d69eb20a8" />




### Data Augmentation
Training Transformations
The training data uses:
- RandomResizedCrop(224)
- RandomHorizontalFlip()
- ImageNet normalization
Validation Transformations
The validation data uses:
- Resize to 224 × 224
- ImageNet normalization
Normalization:
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
### Results
### Baseline Swin-T
| Metric                       |     Result |
| ---------------------------- | ---------: |
| Best Validation Accuracy     | **99.00%** |
| Best Epoch                   |     **26** |
| Final Validation Accuracy    | **98.98%** |
| Maximum Global Gradient Norm | **139.96** |

### Swin-T 5×5 + SiLU
| Metric                       |     Result |
| ---------------------------- | ---------: |
| Best Validation Accuracy     | **98.91%** |
| Best Epoch                   |     **25** |
| Final Validation Accuracy    | **98.89%** |
| Maximum Global Gradient Norm | **126.90** |

The difference between the best validation accuracies is approximately:

98.91% - 99.00% = -0.09 percentage points

Thus, in this single controlled experiment, the modified architecture did not produce a higher validation accuracy than the baseline.

The difference is small, and this experiment does not provide sufficient evidence to claim that the architectural modifications are statistically better or worse across different random seeds.

### Gradient Clipping Analysis
Gradient clipping was applied with a maximum norm of:1.0
The recorded global gradient norms were greater than 1.0, particularly during the early training stages. Therefore, gradient clipping was relevant during optimization.
The maximum recorded global gradient norms were:

Baseline       : 139.96
5×5 + SiLU     : 126.90

Neither experiment exceeded the assignment's threshold of:10³
The gradient magnitudes generally decreased as training progressed.It is important to distinguish between the average per-parameter gradient norm and the global gradient norm. The global norm is the quantity used to determine whether the clipping threshold is exceeded.
### Reproducibility
The experiments use fixed random seeds:
Dataset split seed = 42
Model/training seed = 123

The same train-validation split is used for both the baseline and ablation experiments.

The same training configuration is also maintained between the two experiments to provide a controlled comparison.
Swin_Ablation.ipynb = full implementation, experiments, explanations, visualizations.
run_baseline.py = reproducible baseline run.
run_ablation.py = reproducible modified-model run.
make_plots.py = regenerate the comparison figures.
### Repository Structure
<img width="652" height="802" alt="image" src="https://github.com/user-attachments/assets/a88c72d3-f87e-4190-bd32-ff14a6499a8a" />

### Installation
Clone the repository and install the required dependencies:

- git clone <YOUR-GITHUB-REPOSITORY-URL>
- cd <YOUR-REPOSITORY-NAME>

Install the dependencies:

- pip install -r requirements.txt

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
