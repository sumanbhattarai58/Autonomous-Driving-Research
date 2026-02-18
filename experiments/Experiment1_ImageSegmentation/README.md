# CamVid Semantic Segmentation for Autonomous Driving

Deep learning semantic segmentation pipeline for autonomous driving using CamVid dataset. Features multiple architectures, hyperparameter tuning, and comprehensive evaluation tools.

## Project Overview

This project contains a complete implementation of semantic segmentation for autonomous driving applications, featuring:
- Multiple architecture support (U-Net, DeepLabV3+, etc)
- Class merging strategy to handle severe class imbalance
- Comprehensive data preprocessing pipeline
- Training with weighted loss functions
- Extensive evaluation metrics and visualization tools

## Dataset

**CamVid (Cambridge-driving Labeled Video Database)**
- 367 training images
- Validation and test splits
- Original: 32 classes with severe imbalance
- Processed: 13 merged classes optimized for autonomous driving


## Key Components

### 1. Class Merging Strategy
Problem: Original 32 classes severely imbalanced (some <1% frequency)
Solution: Merged to 13 classes for autonomous driving:

1. **Road** - All road surfaces and lane markings
2. **Sidewalk** - Pedestrian walkable areas
3. **Building** - All structures and walls
4. **Pole** - Street poles and columns
5. **TrafficLight** - Traffic signals
6. **TrafficSign** - All signage and traffic cones
7. **Vegetation** - Trees and plants
8. **Sky** - Sky regions
9. **Person** - Pedestrians and children
10. **Cyclist** - Bicyclists and motorcyclists
11. **Vehicle** - All motor vehicles
12. **Object** - Other moving objects
13. **Ignore** - Background/void class

### 2. Training Pipeline

**Models:** U-Net   
**Encoder** ResNet50   
**Loss:** Weighted Cross-Entropy (median frequency balancing)   
**Augmentation:** Flip, rotate, color jitter, resize   
**Optimization:** Adam/AdamW with ReduceLROnPlateau/Cosine scheduler   
**Early Stopping:** Patience-based validation monitoring(LR_PATIENCE= 5 ,LR_FACTOR = 0.5)   

### 3. Hyperparameter Tuning
**Random search over:**

**Learning rate:** [1e-5, 5e-4, 1e-3]  
**Batch size:** [4, 8]  
**Optimizer:** [Adam, AdamW]  
**Weight decay:** [1e-5, 1e-4, 5e-4]  
**LR scheduler:** [reduce_on_plateau, cosine]  


### 4. Evaluation

Mean IoU and per-class IoU metrics  
Prediction visualization (image, ground truth, prediction)  
Inference speed (FPS) measurement  
MLflow experiment tracking and comparison  

## Results
[IoU metrics](./UnetResults/test_results.txt)  
[Prediction visulization](./UnetResults/predictions_visualization.png)  
[MLflow chart](./UnetResults/mlflow_results.csv)  