# Detailed Summary of the Paper

This summary is based on information from the published paper: [Exploring 3D–2D CNN Feature Hierarchy for Hyperspectral Image Classification](https://arxiv.org/abs/1902.06701)

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Model](#model)
  - [3D CNN Layers](#3d-cnn-layers)
  - [2D CNN Layer](#2d-cnn-layer)
- [Output Shape](#output-shape)
- [Training](#training)
- [Method](#method)
- [Hardware](#hardware)
- [Results](#results)
  - [Benchmarking Tool](#benchmarking-tool)

## Introduction

The paper explores the relationship between 2D-CNN and 3D-CNN architectures for hyperspectral image classification, focusing on the benefits and trade-offs of each approach.

A 2D-CNN is computationally simpler, but it cannot directly capture spectral relationships between bands. In contrast, a 3D-CNN can capture both spatial and spectral information, but it is computationally more expensive and may perform worse for categories with similar textures across many spectral bands.

## Datasets

[Link to all three datasets](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

**Indian Pines (IP)**

- Spatial dimensions: 145 × 145 pixels
- Spectral bands: 224
- Wavelength range: 400 to 2500 nm
- Water absorption bands discarded in the experiment: 24
- Ground-truth classes: 16

**University of Pavia (UP)**

- Spatial dimensions: 610 × 340 pixels
- Spectral bands: 103
- Wavelength range: 430 to 860 nm
- Ground-truth classes: 9

**Salinas Scene (SA)**

- Spatial dimensions: 512 × 217 pixels
- Spectral bands: 224
- Wavelength range: 360 to 2500 nm
- Water absorption bands discarded in the experiment: 20
- Ground-truth classes: 16

## Model

![Model overview](Model_overview.png)

**Input Patch Size and PCA**

For **Indian Pines (IP)** and **Salinas Scene (SA)**:

- Input patch size: 25 × 25
- PCA-reduced spectral bands: 30

For **University of Pavia (UP)**:

- Input patch size: 25 × 25
- PCA-reduced spectral bands: 15

### 3D CNN Layers

The 3D CNN layer structure is described as:

number of filters × kernel height × kernel width × kernel spectral depth × number of input feature maps

**First 3D CNN Layer**
```
8 × 3 × 3 × 7 × 1
```

- Number of filters: 8
- Kernel size: 3 × 3 × 7
- Number of input feature maps: 1

**Second 3D CNN Layer**
```
16 × 3 × 3 × 5 × 8
```

- Number of filters: 16
- Kernel size: 3 × 3 × 5
- Number of input feature maps: 8

**Third 3D CNN Layer**
```
32 × 3 × 3 × 3 × 16
```

- Number of filters: 32
- Kernel size: 3 × 3 × 3
- Number of input feature maps: 16

### 2D CNN Layer

After the 3D CNN layers, the feature maps are reshaped and passed into a 2D CNN layer. The 2D CNN layer structure is described as:
```
number of filters × kernel height × kernel width × number of input feature maps
```

**2D CNN Layer**
```
64 × 3 × 3 × 576
```

- Number of filters: 64
- Kernel size: 3 × 3
- Number of input feature maps: 576

## Output shape

Output shape for IP & SA:
![Output shape](Output_shape.png)

The number of nodes in the final dense layer depends on the number of classes in each dataset:

- **IP**: 16 classes
- **UP**: 9 classes
- **SA**: 16 classes

The total number of trainable parameters in HybridSN is:

- **IP**: 5,122,176
- **UP**: 4,844,793
- **SA**: 4,845,696

## Training

- The dataset is randomly split into **30% training data** and **70% testing data**.
- All weights are randomly initialized.
- The model is trained using the **back-propagation algorithm** with the **Adam optimizer** and **softmax loss**.
- Learning rate: `0.001`
- Mini-batch size: `256`
- Number of epochs: `100`
- No batch normalization or data augmentation is used.

## Method

The authors selected the optimal configuration based on classification performance. To ensure a fair comparison across datasets, they used the same spatial patch size and PCA for the input volume.

The input patch dimensions were:

- **IP**: 25 × 25 × 30
- **UP**: 25 × 25 × 15
- **SA**: 25 × 25 × 15

## Hardware

According to the paper, all experiments were conducted on an Acer Predator Helios laptop with the following specifications:

- GTX 1060 Graphics Processing Unit (GPU)
- 16 GB RAM

## Results

The authors evaluated model performance using the following metrics:

- **Overall Accuracy (OA)**
- **Average Accuracy (AA)**
- **Kappa Coefficient (Kappa)**

The results show that the 3D-CNN performs worse than the 2D-CNN on the Salinas Scene dataset. According to the authors, this may be due to the presence of two classes in the Salinas dataset, **Grapes-untrained** and **Vineyard-untrained**, which have very similar textures across most spectral bands.

### Benchmarking Tool

The authors also refer to the following benchmarking tool:

<https://github.com/eecn/Hyperspectral-Classification>