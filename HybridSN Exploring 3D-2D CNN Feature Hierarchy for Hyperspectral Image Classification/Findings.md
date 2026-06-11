# Findings

## Baseline

There is some differences in author's tensorflow, PyTorch version, and my recreation

### Author's Tensorflow

1. Load the data
2. Apply PCA
- PCA is 30
3. Create patches
4. Split the data into 30 train and 70 test
- Split seed is 345

**Result in Paper**
| Dataset |            OA |            AA |         Kappa |
| ------- | ------------: | ------------: | ------------: |
| IP      |  99.75 ± 0.10 |  99.63 ± 0.20 |  99.71 ± 0.10 |
| UP      |  99.98 ± 0.00 |  99.97 ± 0.00 |  99.98 ± 0.00 |
| SA      | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |

### Pytorch Version

1. Load the data
2. Apply PCA
- PCA is 30
3. Split the data into 30% train, 10% validation and 60% test
- Split seed is 100

4. Create patches

**Result**
| Dataset |            OA |            AA |         Kappa |
| ------- | ------------: | ------------: | ------------: |
| IP      |  99.29 ± 0.56 |  98.51 ± 1.40 |  99.19 ± 0.64 |
| UP      |  99.96 ± 0.02 |  99.91 ± 0.06 |  99.94 ± 0.03 |
| SA      | 100.00 ± 0.00 | 100.00 ± 0.01 | 100.00 ± 0.01 |

Anomalous seed result:

| Dataset | Seed |    OA |    AA | Kappa |
| ------- | ---: | ----: | ----: | ----: |
| IP      |    9 | 23.97 |  6.25 |  0.0  |

### My Recreation

1. Load the data
2. Apply PCA
- PCA is 30
3. Create patches
4. Split the data into 30 train and 70 test
- Split seed is 100

**Result**
| Dataset |           OA |           AA |        Kappa |
| ------- | -----------: | -----------: | -----------: |
| IP      | 99.37 ± 0.35 | 98.27 ± 1.36 | 99.28 ± 0.40 |
| UP      | 99.97 ± 0.03 | 99.91 ± 0.07 | 99.96 ± 0.03 |
| SA      | 99.99 ± 0.01 | 99.99 ± 0.01 | 99.99 ± 0.01 |

To understand the model further let's test it out under different conditions.

## Validation Set

In my experiments, UP and SA still produced relatively stable results even without a validation set, but IP became unstable. In one run without a validation set, the IP result dropped to OA = 90.38, AA = 77.92, and Kappa = 89.03. This suggests that IP is more sensitive to model-selection strategy, likely because of its smaller labelled sample size and stronger class imbalance. Without a validation set, the selected model may simply be the final training epoch, which can overfit to the training data. Alternatively, selecting the best epoch using the test set would make the test result biased. Therefore, using a validation set is important for fair model selection, especially on IP.

Benchmark Script: [benchmark.py](script/benchmark.py) && [without_valid_benchmark.py](script/without_valid_benchmark.py)

**Without validation set**

Split: 30% train and 70% test

| PCA | Dataset |            OA |            AA |         Kappa |
| --- | ------- | ------------: | ------------: | ------------: |
| 30  | IP      |  98.61 ± 2.90 |  96.56 ± 6.60 |  98.42 ± 3.30 |
| 15  | IP      |  99.65 ± 0.11 |  99.51 ± 0.40 |  99.60 ± 0.13 |
| 15  | UP      |  99.97 ± 0.01 |  99.94 ± 0.03 |  99.97 ± 0.01 |
| 15  | SA      | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |

Anomalous seed result:

| PCA | Dataset | Seed |    OA |    AA | Kappa |
| --- | ------- | ---: | ----: | ----: | ----: |
| 30  | IP      |    5 | 90.38 | 77.92 | 89.03 |

**With validation set**

Split: 30% train, 10% validation and 60% test

| PCA | Dataset |           OA |           AA |        Kappa |
| --- | ------- | -----------: | -----------: | -----------: |
| 15  | IP      | 99.60 ± 0.12 | 99.12 ± 0.81 | 99.54 ± 0.14 |
| 30  | IP      | 99.45 ± 0.17 | 98.76 ± 1.01 | 99.37 ± 0.19 |
| 15  | UP      | 99.96 ± 0.03 | 99.90 ± 0.07 | 99.94 ± 0.04 |
| 15  | SA      | 99.99 ± 0.01 | 99.98 ± 0.02 | 99.99 ± 0.01 |

## PCA Sensitivity

How sensitive is HybridSN to spectral compression?

Benchmark Script: [benchmark.py](script/benchmark.py)

| Dataset | PCA |            OA |            AA |         Kappa | Note                                                 |
| ------- | --: | ------------: | ------------: | ------------: | ---------------------------------------------------- |
| IP      |  15 |  99.60 ± 0.12 |  99.12 ± 0.81 |  99.54 ± 0.14 |                                                      |
| IP      |  30 |  99.45 ± 0.17 |  98.76 ± 1.01 |  99.37 ± 0.19 |                                                      |
| IP      |  50 | 92.10 ± 23.76 | 89.91 ± 29.40 | 89.60 ± 31.48 | Large variation due to collapse at seed 9            |
| UP      |  15 |  99.96 ± 0.03 |  99.90 ± 0.07 |  99.94 ± 0.04 |                                                      |
| SA      |  15 |  99.99 ± 0.01 |  99.98 ± 0.02 |  99.99 ± 0.01 |                                                      |

Machine was not powerful enough to run PCA 30 and 50 for both UP and SA.

Anomalous seed result:

| Dataset | PCA | Seed |    OA |   AA | Kappa | Note                       |
| ------- | --: | ---: | ----: | ---: | ----: | -------------------------- |
| IP      |  50 |    9 | 24.48 | 6.25 |  0.00 | Model collapsed completely |

## Patch Size

Patch size can be used to test how much HybridSN benefits from surrounding spatial context. However, changing patch size is not a completely isolated test of spatial dependency, because it also change the model’s feature dimensions and number of parameters, especially before the fully connected layers.

Benchmark Script: [benchmark.py](script/benchmark.py)

| Dataset | Patch Size |           OA |           AA |        Kappa |
| ------- | ---------: | -----------: | -----------: | -----------: |
| IP      |          9 | 97.83 ± 0.35 | 96.54 ± 1.53 | 97.53 ± 0.40 |
| IP      |         15 | 99.53 ± 0.14 | 98.66 ± 0.89 | 99.46 ± 0.16 |
| IP      |         19 | 99.49 ± 0.24 | 98.82 ± 1.01 | 99.42 ± 0.28 |
| IP      |         25 | 99.60 ± 0.12 | 99.12 ± 0.81 | 99.54 ± 0.14 |
| UP      |          9 | 99.86 ± 0.04 | 99.74 ± 0.06 | 99.81 ± 0.05 |
| UP      |         15 | 99.96 ± 0.02 | 99.92 ± 0.04 | 99.95 ± 0.02 |
| UP      |         19 | 99.97 ± 0.02 | 99.93 ± 0.07 | 99.96 ± 0.03 |
| UP      |         25 | 99.96 ± 0.03 | 99.90 ± 0.07 | 99.94 ± 0.04 |
| SA      |          9 | 99.94 ± 0.02 | 99.95 ± 0.02 | 99.93 ± 0.02 |
| SA      |         15 | 99.99 ± 0.01 | 99.99 ± 0.02 | 99.99 ± 0.01 |
| SA      |         19 | 99.99 ± 0.01 | 99.99 ± 0.01 | 99.99 ± 0.01 |
| SA      |         25 | 99.99 ± 0.01 | 99.99 ± 0.01 | 99.99 ± 0.01 |

## Block Split

For block size, the main trade-off is leakage versus class coverage: bigger blocks reduce neighbour leakage, but they can accidentally remove rare classes from train/val/test. Therefore, in the block_split_indices_class_aware method we make sure that the split is being done class wise.

Benchmark Script: [datasplit_benchmark.py](script/datasplit_benchmark.py)

| Dataset |    Block Size |            OA |           AA |         Kappa |
| ------- | ------------: | ------------: | -----------: | ------------: |
| IP      | 1x1 (Control) |  99.60 ± 0.12 | 99.12 ± 0.81 |  99.54 ± 0.14 |
| IP      |           3x3 |  92.99 ± 1.00 | 89.90 ± 2.73 |  91.96 ± 1.13 |
| IP      |           5x5 |  82.30 ± 3.36 | 79.12 ± 5.62 |  79.74 ± 3.86 |
| UP      | 1x1 (Control) |  99.96 ± 0.03 | 99.90 ± 0.07 |  99.94 ± 0.04 |
| UP      |           3x3 |  99.78 ± 0.18 | 99.47 ± 0.51 |  99.71 ± 0.24 |
| UP      |           5x5 |  99.50 ± 0.18 | 98.76 ± 0.52 |  99.33 ± 0.24 |
| UP      |         10x10 |  98.88 ± 0.62 | 97.66 ± 1.30 |  98.50 ± 0.83 |
| UP      |         15x15 |  97.52 ± 0.60 | 95.76 ± 1.12 |  96.68 ± 0.79 |
| UP      |         25x25 |  95.79 ± 2.13 | 93.37 ± 3.14 |  94.22 ± 2.91 |
| UP      |         35x35 |  91.64 ± 2.93 | 88.77 ± 3.44 |  88.51 ± 4.01 |
| UP      |         50x50 | 85.24 ± 10.55 | 85.71 ± 5.90 | 80.33 ± 12.84 |
| UP      |         65x65 |  78.71 ± 7.49 | 83.28 ± 5.50 |  72.57 ± 8.96 |
| UP      |         80x80 | 78.56 ± 12.71 | 80.86 ± 8.38 | 70.74 ± 14.94 |
| SA      | 1x1 (Control) |  99.99 ± 0.01 | 99.98 ± 0.02 |  99.99 ± 0.01 |
| SA      |           3x3 |  99.95 ± 0.04 | 99.93 ± 0.07 |  99.95 ± 0.04 |
| SA      |           5x5 |  99.89 ± 0.11 | 99.85 ± 0.13 |  99.87 ± 0.12 |
| SA      |         10x10 |  99.05 ± 0.40 | 99.34 ± 0.45 |  98.93 ± 0.45 |
| SA      |         15x15 |  97.39 ± 0.93 | 97.97 ± 0.94 |  97.07 ± 1.04 |
| SA      |         25x25 |  92.58 ± 3.01 | 94.15 ± 2.82 |  91.61 ± 3.41 |
| SA      |         35x35 |  87.84 ± 4.64 | 91.74 ± 3.85 |  86.31 ± 5.17 |
| SA      |         50x50 |  78.84 ± 5.64 | 85.89 ± 2.68 |  75.45 ± 6.75 |

## PCA Train Data Only

Applying PCA onto the whole dataset contain a low level of data leak, because PCA has seen the feature distribution of validation/test pixels.
Therefore let's try applying PCA to train data only and fit it to validation and test data.

Benchmark Script: [pca_control_benchmark.py](script/pca_control_benchmark.py)

| Dataset |           OA |           AA |        Kappa |
| ------- | -----------: | -----------: | -----------: |
| IP      | 99.63 ± 0.09 | 98.72 ± 1.06 | 99.57 ± 0.11 |
| UP      | 99.95 ± 0.04 | 99.90 ± 0.08 | 99.94 ± 0.05 |
| SA      | 99.99 ± 0.02 | 99.98 ± 0.03 | 99.99 ± 0.02 |

## Combining Block Split and PCA Train Data Only

Benchmark Script: [pca_control_benchmark.py](script/pca_control_benchmark.py)

| Dataset |    Block Size |            OA |           AA |         Kappa |
| ------- | ------------: | ------------: | -----------: | ------------: |
| IP      | 1x1 (Control) |  99.63 ± 0.09 | 98.72 ± 1.06 |  99.57 ± 0.11 |
| IP      |           3x3 |  93.36 ± 1.01 | 89.96 ± 3.20 |  92.39 ± 1.14 |
| IP      |           5x5 |  84.44 ± 2.28 | 80.07 ± 3.62 |  82.17 ± 2.60 |
| UP      | 1x1 (Control) |  99.95 ± 0.04 | 99.90 ± 0.08 |  99.94 ± 0.05 |
| UP      |           3x3 |  99.82 ± 0.10 | 99.63 ± 0.22 |  99.76 ± 0.13 |
| UP      |           5x5 |  99.66 ± 0.16 | 99.29 ± 0.35 |  99.54 ± 0.21 |
| UP      |         10x10 |  98.65 ± 0.50 | 97.75 ± 1.10 |  98.19 ± 0.67 |
| UP      |         15x15 |  97.65 ± 0.41 | 95.82 ± 0.85 |  96.85 ± 0.56 |
| UP      |         25x25 |  95.70 ± 1.09 | 94.16 ± 0.85 |  94.09 ± 1.45 |
| UP      |         35x35 |  92.41 ± 2.79 | 90.31 ± 3.21 |  89.64 ± 3.66 |
| UP      |         50x50 |  88.76 ± 5.18 | 87.24 ± 5.42 |  84.59 ± 7.22 |
| UP      |         65x65 |  80.78 ± 7.61 | 83.39 ± 4.64 |  75.09 ± 8.25 |
| UP      |         80x80 | 79.67 ± 12.75 | 81.49 ± 8.22 | 72.19 ± 14.42 |
| SA      | 1x1 (Control) |  99.99 ± 0.02 | 99.98 ± 0.03 |  99.99 ± 0.02 |
| SA      |           3x3 |  99.98 ± 0.02 | 99.96 ± 0.03 |  99.97 ± 0.02 |
| SA      |           5x5 |  99.90 ± 0.05 | 99.87 ± 0.13 |  99.89 ± 0.06 |
| SA      |         10x10 |  99.31 ± 0.39 | 99.52 ± 0.31 |  99.23 ± 0.44 |
| SA      |         15x15 |  98.13 ± 1.03 | 98.68 ± 0.82 |  97.90 ± 1.16 |
| SA      |         25x25 |  92.62 ± 2.45 | 94.45 ± 3.45 |  91.66 ± 2.73 |
| SA      |         35x35 |  87.88 ± 4.41 | 92.49 ± 3.62 |  86.37 ± 4.95 |
| SA      |         50x50 |  80.08 ± 5.33 | 86.40 ± 3.19 |  76.85 ± 6.19 |

## Mistakes

The model training became unstable and appeared to collapse toward the dominant class. Most predicted labels were class 10, Soybean-mintill, which is the largest class in the Indian Pines dataset. Now it works, but I was unable to identify the exact cause of this issue.

During the IP experiment, the model showed an unexpected instability during training. Between epoch 56 and epoch 57, the training loss increased sharply from 0.0976 to 1.6173, while the training accuracy dropped from 0.9712 to 0.6244. The model partially recovered in the following epochs and eventually reached a high final accuracy of approximately 0.9902.

```
Saved best model to results/IP/best_model.pth
Epoch [051/100] Train Loss: 0.1299 | Train Acc: 0.9537
Epoch [052/100] Train Loss: 0.1038 | Train Acc: 0.9649
Epoch [053/100] Train Loss: 0.1119 | Train Acc: 0.9668
Epoch [054/100] Train Loss: 0.1008 | Train Acc: 0.9683
Epoch [055/100] Train Loss: 0.0931 | Train Acc: 0.9654
                 Val Loss: 0.0797 | Val Acc: 0.9824
Saved best model to results/IP/best_model.pth
Epoch [056/100] Train Loss: 0.0976 | Train Acc: 0.9712
Epoch [057/100] Train Loss: 1.6173 | Train Acc: 0.6244
Epoch [058/100] Train Loss: 1.4698 | Train Acc: 0.5449
Epoch [059/100] Train Loss: 0.9492 | Train Acc: 0.6971
Epoch [060/100] Train Loss: 0.7465 | Train Acc: 0.7415
                 Val Loss: 0.4902 | Val Acc: 0.8535
Epoch [061/100] Train Loss: 0.5675 | Train Acc: 0.8244
Epoch [062/100] Train Loss: 0.4262 | Train Acc: 0.8600
Epoch [063/100] Train Loss: 0.3432 | Train Acc: 0.8907
Epoch [064/100] Train Loss: 0.2723 | Train Acc: 0.8990
Epoch [065/100] Train Loss: 0.2097 | Train Acc: 0.9307
                 Val Loss: 0.1609 | Val Acc: 0.9531
```

The issue of unstable training and having lower performance was from the leftover code that I put to try to solve an issue along time ago.

```
X_pca = X_pca.astype(np.float32)
mins = X_pca.min(axis=(0, 1), keepdims=True)
maxs = X_pca.max(axis=(0, 1), keepdims=True)
X_pca = (X_pca - mins) / (maxs - mins + 1e-8)
```