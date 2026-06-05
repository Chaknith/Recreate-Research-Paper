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

**Result**

OA: 99.81

### Pytorch Version

1. Load the data
2. Apply PCA
- PCA is 30
3. Split the data into 30% train, 10% validation and 60% test
- Split seed is 100

4. Create patches

**Result**
IP
OA: 99.41
AA: 99.32
Kappa: 99.33

UP
OA: 99.93
AA: 99.87
Kappa: 99.90

SA
OA: 99.99
AA: 99.98
Kappa: 99.99

### My Recreation

1. Load the data
2. Apply PCA
- PCA is 30
3. Create patches
4. Split the data into 30 train and 70 test
- Split seed is 100

**Result**
IP
OA: 99.53
AA: 99.34
Kappa: 99.46

UP
OA: 99.97
AA: 99.92
Kappa: 99.96

SA
OA: 100.00
AA: 99.99
Kappa: 100.00

To understand the model further let's test it out under different conditions.

## Validation Set

In my experiments, UP and SA still produced relatively stable results even without a validation set, but IP became unstable. In one run without a validation set, the IP result dropped to OA = 90.38, AA = 77.92, and Kappa = 89.03. This suggests that IP is more sensitive to model-selection strategy, likely because of its smaller labelled sample size and stronger class imbalance. Without a validation set, the selected model may simply be the final training epoch, which can overfit to the training data. Alternatively, selecting the best epoch using the test set would make the test result biased. Therefore, using a validation set is important for fair model selection, especially on IP.

Benchmark Script: [benchmark.py](script/benchmark.py) && [without_valid_benchmark.py](script/without_valid_benchmark.py)

**Without validation set**

Split: 30% train and 70% test

PCA 30
IP
OA: 98.61 ± 2.90
AA: 96.56 ± 6.60
Kappa: 98.42 ± 3.30

At seed 5 the accuracy went down significantly
OA: 90.38
AA: 77.92
Kappa: 89.03

PCA 15
IP
OA: 99.65 ± 0.11
AA: 99.51 ± 0.40
Kappa: 99.60 ± 0.13

UP
OA: 99.97 ± 0.01
AA: 99.94 ± 0.03
Kappa: 99.97 ± 0.01

SA
OA: 100.00 ± 0.00
AA: 100.00 ± 0.00
Kappa: 100.00 ± 0.00

**With validation set**

Split: 30% train, 10% validation and 60% test

PCA 30
IP
OA: 99.45 ± 0.17
AA: 98.76 ± 1.01
Kappa: 99.37 ± 0.19

PCA 15
IP
OA: 99.60 ± 0.12
AA: 99.12 ± 0.81
Kappa: 99.54 ± 0.14

UP
OA: 99.96 ± 0.03
AA: 99.90 ± 0.07
Kappa: 99.94 ± 0.04

SA
OA: 99.99 ± 0.01
AA: 99.98 ± 0.02
Kappa: 99.99 ± 0.01

## PCA Sensitivity

How sensitive is HybridSN to spectral compression?

Benchmark Script: [benchmark.py](script/benchmark.py)

IP
PCA 15
OA: 99.60 ± 0.12
AA: 99.12 ± 0.81
Kappa: 99.54 ± 0.14

PCA 30
OA: 99.45 ± 0.17
AA: 98.76 ± 1.01
Kappa: 99.37 ± 0.19

PCA 50
OA: 92.10 ± 23.76
AA: 89.91 ± 29.40
Kappa: 89.60 ± 31.48

In seed 9 the model complete collapse
OA: 24.48
AA: 06.25
Kappa: 00.00

UP
PCA 15
OA: 99.96 ± 0.03
AA: 99.90 ± 0.07
Kappa: 99.94 ± 0.04

PCA 30
Machine is not powerful enough to run the test with PCA 30 and 50

SA
PCA 15
OA: 99.99 ± 0.01
AA: 99.98 ± 0.02
Kappa: 99.99 ± 0.01

PCA 30
Machine is not powerful enough to run the test with PCA 30 and 50

## Patch Size

Patch size can be used to test how much HybridSN benefits from surrounding spatial context. However, changing patch size is not a completely isolated test of spatial dependency, because it also change the model’s feature dimensions and number of parameters, especially before the fully connected layers.

Benchmark Script: [benchmark.py](script/benchmark.py)

IP
Patch 9
OA: 97.83 ± 0.35
AA: 96.54 ± 1.53
Kappa: 97.53 ± 0.40

Patch 15
OA: 99.53 ± 0.14
AA: 98.66 ± 0.89
Kappa: 99.46 ± 0.16

Patch 19
OA: 99.49 ± 0.24
AA: 98.82 ± 1.01
Kappa: 99.42 ± 0.28

Patch 25
OA: 99.60 ± 0.12
AA: 99.12 ± 0.81
Kappa: 99.54 ± 0.14

UP
Patch 9
OA: 99.86 ± 0.04
AA: 99.74 ± 0.06
Kappa: 99.81 ± 0.05

Patch 15
OA: 99.96 ± 0.02
AA: 99.92 ± 0.04
Kappa: 99.95 ± 0.02

Patch 19
OA: 99.97 ± 0.02
AA: 99.93 ± 0.07
Kappa: 99.96 ± 0.03

Patch 25
OA: 99.96 ± 0.03
AA: 99.90 ± 0.07
Kappa: 99.94 ± 0.04

SA
Patch 9
OA: 99.94 ± 0.02
AA: 99.95 ± 0.02
Kappa: 99.93 ± 0.02

Patch 15
OA: 99.99 ± 0.01
AA: 99.99 ± 0.02
Kappa: 99.99 ± 0.01

Patch 19
OA: 99.99 ± 0.01
AA: 99.99 ± 0.01
Kappa: 99.99 ± 0.01

Patch 25
OA: 99.99 ± 0.01
AA: 99.99 ± 0.01
Kappa: 99.99 ± 0.01

## Block Split

For block size, the main trade-off is leakage versus class coverage: bigger blocks reduce neighbour leakage, but they can accidentally remove rare classes from train/val/test. Therefore, in the block_split_indices_class_aware method we make sure that the split is being done class wise.

Benchmark Script: [datasplit_benchmark.py](script/datasplit_benchmark.py)

PCA 15
Block Size 1x1(Control)
IP
OA: 99.60 ± 0.12
AA: 99.12 ± 0.81
Kappa: 99.54 ± 0.14

UP
OA: 99.96 ± 0.03
AA: 99.90 ± 0.07
Kappa: 99.94 ± 0.04

SA
OA: 99.99 ± 0.01
AA: 99.98 ± 0.02
Kappa: 99.99 ± 0.01

Block Size 3x3
IP
OA: 92.99 ± 1.00
AA: 89.90 ± 2.73
Kappa: 91.96 ± 1.13

UP
OA: 99.78 ± 0.18
AA: 99.47 ± 0.51
Kappa: 99.71 ± 0.24

SA
OA: 99.95 ± 0.04
AA: 99.93 ± 0.07
Kappa: 99.95 ± 0.04

Block Size 5x5
IP
OA: 82.30 ± 3.36
AA: 79.12 ± 5.62
Kappa: 79.74 ± 3.86

UP
OA: 99.50 ± 0.18
AA: 98.76 ± 0.52
Kappa: 99.33 ± 0.24

SA
OA: 99.89 ± 0.11
AA: 99.85 ± 0.13
Kappa: 99.87 ± 0.12

10x10
UP
OA: 98.88 ± 0.62
AA: 97.66 ± 1.30
Kappa: 98.50 ± 0.83

SA
OA: 99.05 ± 0.40
AA: 99.34 ± 0.45
Kappa: 98.93 ± 0.45

15x15
UP
OA: 97.52 ± 0.60
AA: 95.76 ± 1.12
Kappa: 96.68 ± 0.79

SA
OA: 97.39 ± 0.93
AA: 97.97 ± 0.94
Kappa: 97.07 ± 1.04

25x25
UP
OA: 95.79 ± 2.13
AA: 93.37 ± 3.14
Kappa: 94.22 ± 2.91

SA
OA: 92.58 ± 3.01
AA: 94.15 ± 2.82
Kappa: 91.61 ± 3.41

35x35
UP
OA: 91.64 ± 2.93
AA: 88.77 ± 3.44
Kappa: 88.51 ± 4.01

SA
OA: 87.84 ± 4.64
AA: 91.74 ± 3.85
Kappa: 86.31 ± 5.17

50x50
UP
OA: 85.24 ± 10.55
AA: 85.71 ± 5.90
Kappa: 80.33 ± 12.84

SA
OA: 78.84 ± 5.64
AA: 85.89 ± 2.68
Kappa: 75.45 ± 6.75

65x65
UP
OA: 78.71 ± 7.49
AA: 83.28 ± 5.50
Kappa: 72.57 ± 8.96

80x80
UP
OA: 78.56 ± 12.71
AA: 80.86 ± 8.38
Kappa: 70.74 ± 14.94

## PCA Train Data Only

Applying PCA onto the whole dataset contain a low level of data leak, because PCA has seen the feature distribution of validation/test pixels.
Therefore let's try applying PCA to train data only and fit it to validation and test data.

Benchmark Script: [pca_control_benchmark.py](script/pca_control_benchmark.py)

IP
OA: 99.63 ± 0.09
AA: 98.72 ± 1.06
Kappa: 99.57 ± 0.11

UP
OA: 99.95 ± 0.04
AA: 99.90 ± 0.08
Kappa: 99.94 ± 0.05

SA
OA: 99.99 ± 0.02
AA: 99.98 ± 0.03
Kappa: 99.99 ± 0.02

## Combining Block Split and PCA Train Data Only

Benchmark Script: [pca_control_benchmark.py](script/pca_control_benchmark.py)

Block size 1x1(control)
IP
OA: 99.63 ± 0.09
AA: 98.72 ± 1.06
Kappa: 99.57 ± 0.11

UP
OA: 99.95 ± 0.04
AA: 99.90 ± 0.08
Kappa: 99.94 ± 0.05

SA
OA: 99.99 ± 0.02
AA: 99.98 ± 0.03
Kappa: 99.99 ± 0.02

Block size 3x3
IP
OA: 93.36 ± 1.01
AA: 89.96 ± 3.20
Kappa: 92.39 ± 1.14

UP
OA: 99.82 ± 0.10
AA: 99.63 ± 0.22
Kappa: 99.76 ± 0.13

SA
OA: 99.98 ± 0.02
AA: 99.96 ± 0.03
Kappa: 99.97 ± 0.02

Block size 5x5
IP
OA: 84.44 ± 2.28
AA: 80.07 ± 3.62
Kappa: 82.17 ± 2.60

UP
OA: 99.66 ± 0.16
AA: 99.29 ± 0.35
Kappa: 99.54 ± 0.21

SA
OA: 99.90 ± 0.05
AA: 99.87 ± 0.13
Kappa: 99.89 ± 0.06

Block size 10x10
UP
OA: 98.65 ± 0.50
AA: 97.75 ± 1.10
Kappa: 98.19 ± 0.67

SA
OA: 99.31 ± 0.39
AA: 99.52 ± 0.31
Kappa: 99.23 ± 0.44

Block size 15x15
UP
OA: 97.65 ± 0.41
AA: 95.82 ± 0.85
Kappa: 96.85 ± 0.56

SA
OA: 98.13 ± 1.03
AA: 98.68 ± 0.82
Kappa: 97.90 ± 1.16

Block size 25x25
UP
OA: 95.70 ± 1.09
AA: 94.16 ± 0.85
Kappa: 94.09 ± 1.45

SA
OA: 92.62 ± 2.45
AA: 94.45 ± 3.45
Kappa: 91.66 ± 2.73

Block size 35x35
UP
OA: 92.41 ± 2.79
AA: 90.31 ± 3.21
Kappa: 89.64 ± 3.66

SA
OA: 87.88 ± 4.41
AA: 92.49 ± 3.62
Kappa: 86.37 ± 4.95

Block size 50x50
UP
OA: 88.76 ± 5.18
AA: 87.24 ± 5.42
Kappa: 84.59 ± 7.22

SA
OA: 80.08 ± 5.33
AA: 86.40 ± 3.19
Kappa: 76.85 ± 6.19

Block size 65x65
UP
OA: 80.78 ± 7.61
AA: 83.39 ± 4.64
Kappa: 75.09 ± 8.25

Block size 80x80
UP
OA: 79.67 ± 12.75
AA: 81.49 ± 8.22
Kappa: 72.19 ± 14.42

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