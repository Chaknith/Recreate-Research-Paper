from helper import DATASETS, load_hsi_data, apply_pca, create_patches, HSIDataset, HybridSN, train_one_epoch, evaluate, predict_loader
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import spectral
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score, classification_report
from tqdm import tqdm
from typing import Dict

# Set hypeperameters and experimental settings just like the original code
seeds = [1,2,3,4,5,6,7,8,9,10]
DATASET = 'IP'      # IP, UP, SA  
TRAIN_SIZE = 0.3    # ratio of training data
VAL_SIZE = 0.1      # ratio of valuating data
EPOCH = 100         # number of epoch
VAL_EPOCH = 5       # interval of valuation
LR = 0.001          # learning rate
WEIGHT_DECAY = 1e-6  
BATCH_SIZE = 256
DEVICE = 0          # -1:CPU  0:cuda 0
N_PCA = 15
PATCH_SIZE = 25
DROP_OUT = 0.4
class_name = DATASETS[DATASET]["class_name"]
NUM_CLASS = len(class_name)
save_dir = Path("results") / DATASET
save_dir.mkdir(parents=True, exist_ok=True)

config_comment = f"""# Configuration
# seeds = {seeds}
# DATASET = {DATASET!r}      # IP, UP, SA
# TRAIN_SIZE = {TRAIN_SIZE}    # ratio of training data
# VAL_SIZE = {VAL_SIZE}      # ratio of validation data
# EPOCH = {EPOCH}         # number of epochs
# VAL_EPOCH = {VAL_EPOCH}       # interval of validation
# LR = {LR}             # learning rate
# WEIGHT_DECAY = {WEIGHT_DECAY}
# BATCH_SIZE = {BATCH_SIZE}
# DEVICE = {DEVICE}          # -1: CPU, 0: cuda:0
# N_PCA = {N_PCA}
# PATCH_SIZE = {PATCH_SIZE}
# DROP_OUT = {DROP_OUT}
"""

benchmark_results = []
training_data = []
validation_data = []

benchmark_results_name = f"results/{DATASET}_benchmark_results.py"
training_data_name = f"results/{DATASET}_testing_data_report.py"
validation_data_name = f"results/{DATASET}_validation_data_report.py"

X, y = load_hsi_data(DATASET)
# print(X.shape)  # e.g. (145, 145, 200)
# print(y.shape)  # e.g. (145, 145)

for seed in seeds:
    print(f"Running seed {seed}...")
    # Set random seed just like the original code
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Apply PCA
    X_pca = apply_pca(X, n_components=N_PCA)
    # print(X_pca.shape)  # (145, 145, 30)

    # Create Patches
    X_all, y_all = create_patches(X_pca, y, PATCH_SIZE, True)

    # Get test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all,
        y_all,
        train_size=TRAIN_SIZE,
        random_state=seed,
        stratify=y_all
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=VAL_SIZE / (1-TRAIN_SIZE),
        random_state=seed,
        stratify=y_temp
    )

    train_loader = DataLoader(HSIDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(HSIDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(HSIDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device(f"cuda:{DEVICE}" if DEVICE >= 0 and torch.cuda.is_available() else "cpu")

    num_classes = len(DATASETS[DATASET]["class_name"])
    model = HybridSN(
        num_classes,
        N_PCA,
        PATCH_SIZE,
        dropout=DROP_OUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    best_model_path = save_dir / f"best_model_{DATASET}_iteration{seed}.pth"

    data1 = []
    data2 = []
    for epoch in tqdm(range(1, EPOCH + 1)):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        data1.append({
            "train_loss": train_loss,
            "train_acc": train_acc,
        })
        # print(
        #     f"Epoch [{epoch:03d}/{EPOCH}] "
        #     f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
        # )

        if epoch % VAL_EPOCH == 0:
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device
            )
            data2.append({
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

            # print(
            #     f"                 "
            #     f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
            # )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                #print(f"Saved best model to {best_model_path}")

    training_data.append({
        "seed": seed,
        "data": data1
    })
    validation_data.append({
        "seed": seed,
        "data": data2
    })

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_true, test_pred = predict_loader(model, test_loader, device)

    OA = accuracy_score(test_true, test_pred)
    AA = recall_score(test_true, test_pred, average="macro")
    kappa = cohen_kappa_score(test_true, test_pred)
    benchmark_results.append({
        "seed": seed,
        "OA": OA,
        "AA": AA,
        "Kappa": kappa
    })

    report_log = f"OA: {OA:.4f}\nAA: {AA:.4f}\nKappa: {kappa:.4f}\n"
    # report_log += classification_report(
    #     test_true,
    #     test_pred,
    #     labels=np.arange(NUM_CLASS),
    #     target_names=class_name,
    #     digits=4
    # )

    print(report_log)

## Save all the data into text file
with open(os.path.join(benchmark_results_name), "w") as fp:
    fp.write("# Auto-generated benchmark results\n")
    fp.write(config_comment + "\n")
    fp.write("results = ")
    fp.write(repr(benchmark_results))
    fp.write("\n")
with open(os.path.join(training_data_name), "w") as fp:
    fp.write("# Auto-generated testing data\n")
    fp.write(config_comment + "\n")
    fp.write("testing_data = ")
    fp.write(repr(training_data))
    fp.write("\n")
with open(os.path.join(validation_data_name), "w") as fp:
    fp.write("# Auto-generated validation data\n")
    fp.write(config_comment + "\n")
    fp.write("validation_data = ")
    fp.write(repr(validation_data))
    fp.write("\n")