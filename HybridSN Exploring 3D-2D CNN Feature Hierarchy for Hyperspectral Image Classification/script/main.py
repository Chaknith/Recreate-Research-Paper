from helper import DATASETS, load_hsi_data, apply_pca
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import numpy as np
import os
import random

# Set hypeperameters and experimental settings just like the original code
RANDOM_SEED=666
DATASET = 'IP'      # IP, PU, SA  
TRAIN_RATE = 0.3    # ratio of training data
VAL_RATE = 0.1      # ratio of valuating data
EPOCH = 100         # number of epoch
VAL_EPOCH = 5       # interval of valuation
LR = 0.001          # learning rate
WEIGHT_DECAY = 1e-6  
BATCH_SIZE = 256
DEVICE = 0          # -1:CPU  0:cuda 0
N_PCA = 15          # reserved PCA components
PATCH_SIZE = 25 
SAVE_PATH = f"results\{DATASET}"
if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

# Set random seed just like the original code
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

X, y = load_hsi_data(DATASET)

print(X.shape)  # e.g. (145, 145, 200)
print(y.shape)  # e.g. (145, 145)

# # 30 for IP & SA
# # 15 for UP
# X_pca = apply_pca(X, n_components=30)   # for Indian Pines
# print(X.shape)      # e.g. (145, 145, 200)
# print(X_pca.shape)  #       (145, 145, 30)

# X_pca = apply_pca(X, n_components=15)