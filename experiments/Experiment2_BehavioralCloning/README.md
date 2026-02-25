# Behavioral Cloning Steering Angle Regression for Autonomous Driving
Deep learning behavioral cloning pipeline for autonomous driving using Udacity Self-Driving Car dataset. Features NVIDIA end-to-end architecture, hyperparameter tuning, and comprehensive evaluation tools.

---

## Project Overview
This project contains a complete implementation of steering angle regression for autonomous driving applications, featuring:

- NVIDIA End-to-End Learning architecture
- Three camera input strategy (center, left, right) with steering correction
- Comprehensive data preprocessing and augmentation pipeline
- Training with histogram-based steering distribution balancing
- Randomized hyperparameter search with early stopping
- Extensive evaluation metrics and visualization tools

---

## Dataset
**Udacity Self-Driving Car Behavioural Cloning Dataset (Kaggle)**

- Center, left, and right camera images
- Steering angle labels per frame
- Single `driving_log.csv` with all image paths and labels
- Severe zero-steering imbalance addressed via histogram balancing

---

## Key Components

### 1. Data Preprocessing Pipeline
- **Camera selection**: Randomly pick center, left, or right camera during training
- **Steering correction**: ±0.2 applied to left/right cameras
- **Cropping**: Remove sky (top 40px) and hood (bottom 20px)
- **Resize**: To NVIDIA input size (66 x 200)
- **Color space**: RGB → YUV (as per NVIDIA paper)
- **Normalization**: Pixel values scaled to [-1, 1]

### 2. Data Balancing Strategy
**Problem**: Majority of steering angles are near-zero (straight driving bias)

**Solution**:
- Zero steering reduction — keeps only 30% of near-zero samples
- Histogram-based balancing — downsample overrepresented steering bins to mean count
- Optional WeightedRandomSampler for further online balancing

### 3. Augmentation Pipeline (Training Only)
- Random horizontal flip (steering angle negated accordingly)  
- Random brightness and contrast adjustment  
- Random shadow simulation  
- Gaussian blur  
- Random resized crop jitter  

## Training Pipeline

### Model Experiments

### Model 1 — NVIDIA Baseline (as per paper)
- **Activation**: ELU
- **Optimizer**: Adam
- **Dropout**: 0.0
- **Learning rate**: 1e-4
- **Batch size**: 32
- **epochs**: 20
- Follows original NVIDIA End-to-End paper architecture exactly

### Model 2 — Modified Architecture
- **Hyperparameter tuning**: Randomized Search
- **Activation**: ReLU
- **Optimizer**: AdamW
- **Dropout**: [0.3, 0.5]
- **Learning rate**: [1e-4, 3e-4, 1e-3, 3e-3]
- **Batch size**: [16, 32]
- Architecture modifications to compare against baseline

### Comparison
| | Model 1 (NVIDIA) | Model 2:Best Params (Modified) |
|---|---|---|
| MSE | 0.212299 | 0.159969 |
| RMSE | 0.460759 | 0.399961 |
| MAE | 0.345905 | 0.302049 |

**Model 2 best parameters**:
learning_rate: 0.001
weight_decay: 1e-06
batch_size: 32
dropout: 0.3

---

## Project Structure

```
Experiment2_Behavioralcloning/
├── dataset.py          — dataset class, augmentations, balancing, dataloaders
├── model.py            — NVIDIA CNN architecture
├── train.py            — training loop
├── evaluate.py         — metrics, loss curves, prediction plots
├── inference.py        — predict steering angle on new images
├── checkpoints/        — saved model checkpoints (.pt files)
├── Results/            — Results of Nvidia model
│   ├── losses.txt               — train/val loss per epoch per trial
│   ├── best_checkpoint.txt      — path to best saved model
│   ├── evaluation_metrics.txt   — MSE, RMSE, MAE
│   ├── evaluation_plot.png      — predicted vs actual plot
│   └── training_validation_loss.png
└── notebooks/
    └── data_exploration.ipynb   — dataset visualization and exploration
└── Nvidia_tuning/
    └── model.py                 — Modified CNN architecture
    └── train.py                 — training loop, randomized search, early stopping
    └── evaluate.py              — metrics, loss curves, prediction plots
    └── inference.py             — predict steering angle on new images
    └── checkpoints/             — saved model checkpoints (.pt files)
    └── Results/                 — Results of Modified model
       ├── losses.txt               — train/val loss per epoch per trial
       ├── best_checkpoint.txt      — path to best saved model
       ├── best_hyperparams.txt     — best hyperparameter values
       ├── evaluation_metrics.txt   — MSE, RMSE, MAE
       ├── evaluation_plot.png      — predicted vs actual plot
       └── training_validation_loss.png
└──README.md
```

---

## Results
- **Loss metrics (MSE, RMSE, MAE)**
 [model1_losses](Results/losses.txt)  
 [model2_losses](Nvidia_tuning/Results/losses.txt)  

 - **Inference on test images**
  [model1_predictions](Results/inference_result.png) [Prediction_Value](Results/inference_result.txt)  
  [model2_predictions](Nvidia_tuning/Results/inference_result.png) [Prediction_Value](Nvidia_tuning/Results/inference_result.txt)  

- **Predicted vs actual steering angle visualization**
 [model1_steering_angle_visualization](Results/evaluation_plot.png)  
 [model2_steering_angle_visualization](Nvidia_tuning/Results/evaluation_plot.png)  

- **Training and validation loss curves**
 [model1_training_validation_losses](Results/training_validation_loss.png)  
 [model2_training_validation_losses](Nvidia_tuning/Results/training_validation_loss.png)  

- **Evaluation metrics**
 [model1_evaluation_metrics](Results/evaluation_metrics.txt)  
 [model2_evaluation_metrics](Nvidia_tuning/Results/evaluation_metrics.txt)  

-**Training checkpoints**
 [model1_best_checkpoint](Results/best_checkpoint.txt)  
 [model2_best_checkpoint](Nvidia_tuning/Results/best_checkpoint.txt)  
 [model2_best _hyperparameters](Nvidia_tuning/Results/best_hyperparams.txt)  