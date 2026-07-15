# Findings

Actually, since this does not use patches we can just do a pixel split of the data.

## Apply PCA 15

- RANDOM_SEED = 42
- With PCA 15 the performance on test set are below.

Overall Accuracy: 0.1093
Average Accuracy: 0.0325
Kappa: -0.1169
| Class                | Precision |   Recall | F1-score |    Support |
| -------------------- | --------: | -------: | -------: | ---------: |
| Asphalt              |      0.00 |     0.00 |     0.00 |      2,614 |
| Meadows              |      0.40 |     0.18 |     0.25 |     36,234 |
| Trees                |      0.02 |     0.05 |     0.02 |      6,429 |
| Bare Soil            |      0.00 |     0.00 |     0.00 |      2,422 |
| Bitumen              |      0.00 |     0.00 |     0.00 |      5,571 |
| Self-Blocking Bricks |      0.00 |     0.00 |     0.00 |      2,272 |
| Shadows              |      0.00 |     0.00 |     0.00 |      6,165 |
| **Accuracy**         |           |          | **0.11** | **61,707** |
| **Macro avg**        |  **0.06** | **0.03** | **0.04** | **61,707** |
| **Weighted avg**     |  **0.24** | **0.11** | **0.15** | **61,707** |

With this configurations, it seems like the model could not learn.

## Did not apply PCA

Without PCA the result is much better, but it is still low.

Overall Accuracy: 0.6359
Average Accuracy: 0.2311
Kappa: 0.3539
| Class                | Precision |   Recall | F1-score |    Support |
| -------------------- | --------: | -------: | -------: | ---------: |
| Asphalt              |      0.00 |     0.00 |     0.00 |      2,614 |
| Meadows              |      0.80 |     0.97 |     0.88 |     36,234 |
| Trees                |      0.98 |     0.65 |     0.78 |      6,429 |
| Bare Soil            |      0.00 |     0.00 |     0.00 |      2,422 |
| Bitumen              |      0.00 |     0.00 |     0.00 |      5,571 |
| Self-Blocking Bricks |      0.00 |     0.00 |     0.00 |      2,272 |
| Shadows              |      0.00 |     0.00 |     0.00 |      6,165 |
| **Accuracy**         |           |          | **0.64** | **61,707** |
| **Macro avg**        |  **0.25** | **0.23** | **0.24** | **61,707** |
| **Weighted avg**     |  **0.57** | **0.64** | **0.60** | **61,707** |


however, base on the training data it looks like the model overfitted.
```
Epoch 001/100 | Train Loss: 3.8532 | Train Acc: 0.5504 | Val Loss: 2.3398 | Val Acc: 0.5978
Saved new best model: 0.5978
Epoch 002/100 | Train Loss: 0.8103 | Train Acc: 0.6765 | Val Loss: 2.4347 | Val Acc: 0.6302
Saved new best model: 0.6302
Epoch 003/100 | Train Loss: 0.7464 | Train Acc: 0.7058 | Val Loss: 2.7988 | Val Acc: 0.5894
Epoch 004/100 | Train Loss: 0.7073 | Train Acc: 0.7232 | Val Loss: 3.2625 | Val Acc: 0.6165
Epoch 005/100 | Train Loss: 0.6713 | Train Acc: 0.7405 | Val Loss: 3.4803 | Val Acc: 0.6337
Saved new best model: 0.6337
Epoch 006/100 | Train Loss: 0.6525 | Train Acc: 0.7493 | Val Loss: 3.6432 | Val Acc: 0.5043
Epoch 007/100 | Train Loss: 0.6371 | Train Acc: 0.7540 | Val Loss: 3.4815 | Val Acc: 0.5754
Epoch 008/100 | Train Loss: 0.5983 | Train Acc: 0.7679 | Val Loss: 4.1269 | Val Acc: 0.4796
Epoch 009/100 | Train Loss: 0.5829 | Train Acc: 0.7767 | Val Loss: 4.4257 | Val Acc: 0.5305
Epoch 010/100 | Train Loss: 0.5733 | Train Acc: 0.7784 | Val Loss: 4.3652 | Val Acc: 0.4164
Epoch 011/100 | Train Loss: 0.5502 | Train Acc: 0.7917 | Val Loss: 4.1184 | Val Acc: 0.4734
Epoch 012/100 | Train Loss: 0.5160 | Train Acc: 0.8054 | Val Loss: 4.5856 | Val Acc: 0.3072
Epoch 013/100 | Train Loss: 0.5060 | Train Acc: 0.8074 | Val Loss: 5.4537 | Val Acc: 0.1589
Epoch 014/100 | Train Loss: 0.5141 | Train Acc: 0.8028 | Val Loss: 4.6885 | Val Acc: 0.2558
Epoch 015/100 | Train Loss: 0.4894 | Train Acc: 0.8125 | Val Loss: 4.8721 | Val Acc: 0.3065
Epoch 016/100 | Train Loss: 0.4678 | Train Acc: 0.8227 | Val Loss: 5.8435 | Val Acc: 0.1292
Epoch 017/100 | Train Loss: 0.4626 | Train Acc: 0.8271 | Val Loss: 5.8973 | Val Acc: 0.1144
Epoch 018/100 | Train Loss: 0.4569 | Train Acc: 0.8269 | Val Loss: 5.9227 | Val Acc: 0.1503
Epoch 019/100 | Train Loss: 0.4325 | Train Acc: 0.8393 | Val Loss: 5.8553 | Val Acc: 0.1044
Epoch 020/100 | Train Loss: 0.4383 | Train Acc: 0.8367 | Val Loss: 6.4688 | Val Acc: 0.1010
Epoch 021/100 | Train Loss: 0.4172 | Train Acc: 0.8451 | Val Loss: 6.2052 | Val Acc: 0.1031
Epoch 022/100 | Train Loss: 0.4111 | Train Acc: 0.8484 | Val Loss: 5.7526 | Val Acc: 0.1028
Epoch 023/100 | Train Loss: 0.4084 | Train Acc: 0.8472 | Val Loss: 6.6540 | Val Acc: 0.1023
Epoch 024/100 | Train Loss: 0.3949 | Train Acc: 0.8538 | Val Loss: 6.6348 | Val Acc: 0.1075
```

Also the model only perdict Meadows and Trees classes.
I'm not sure how to make the model generalize across different classes.

Try adding SVM to the last step, but it still does not help.