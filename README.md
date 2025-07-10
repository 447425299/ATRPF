# ATRPF

This repository contains the official implementation of the paper "UAV-Satellite Cross-view Image Matching Based on Adaptive Threshold-guided Ring Partitioning Framework" (submitted to Remote Sensing). The ATRPF framework addresses the challenges of cross-domain image matching between UAV and satellite platforms, achieving state-of-the-art performance on the University-1652 benchmark.

## Framework Overview

ATRPF introduces an innovative approach to bridge the domain gap between UAV and satellite imagery, comprising three core components:
- **Adaptive Ring Partitioning**: Dynamically adjusts feature extraction regions using heatmap-guided thresholds and learnable hyperparameters.
- **Brightness Alignment**: Normalizes UAV image brightness to satellite references, reducing illumination-induced variations.
- **Keypoint-aware Re-ranking**: Refines retrieval results using geometrically consistent keypoint matches (SP+SG).


## Environment Setup

### Dependencies
- Python 3.6+
- PyTorch 1.8.0+
- torchvision 0.9.0+

### Installation
```bash
git clone https://github.com/447425299/ATRPF.git
cd ATRPF
```

## Dataset Preparation

### University-1652 Dataset
1. Download the dataset from: https://github.com/layumi/University1652-Baseline
2. Organize the dataset as follows:
```
data/
├── University1652/
│   ├── satellite_images/
│   ├── uav_images/
│   ├── street_view_images/
│   └── google_images/
```

## Training & Testing

### Training the ATRPF Model
```
python train.py --name three_view_long_share_d0.75_256_s1_google  --extra --views 3  --droprate 0.75  --share  --stride 1 --h 256  --w 256 --fp16; 
python test.py --name three_view_long_share_d0.75_256_s1_google
```


## Evaluation Metrics
- **Recall@K**: Proportion of queries where the ground truth is within the top K results.
- **Average Precision (AP)**: Area under the Precision-Recall curve, balancing precision and recall.

## Performance Comparison

| University-1652 | UAV→Satellite (Recall@1 / AP) | Satellite→UAV (Recall@1 / AP) |
|--------|------------------------------|------------------------------|
| ATRPF  | **82.50 / 84.28**          | **90.87 / 80.25**          |

## Pre-trained Models

You can download the pre-trained ATRPF model weights from the following link:

**Download Link**: [Quark Drive]([https://pan.quark.cn/s/0a44fcab7cb2])  
**Password**: JcjG
