import numpy as np
from pathlib import Path
from typing import Dict
from scipy.io import loadmat
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

DATASETS: Dict[str, Dict[str, object]] = {
    "IP": {
        "folder": "IP",
        "description": "Indian Pines (corrected)",
        "data_file": "Indian_pines_corrected.mat",
        "gt_file": "Indian_pines_gt.mat",
        "data_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
        "class_name": ['Alfalfa','Corn-notill','Corn-mintill','Corn','Grass-pasture',
        'Grass-trees','Grass-pasture-mowed','Hay-windrowed','Oats','Soybean-notill',
        'Soybean-mintill','Soybean-clean','Wheat','Woods','Buildings-Grass-Trees-Drives',
        'Stone-Steel-Towers'],
    },
    "UP": {
        "folder": "UP",
        "description": "Pavia University",
        "data_file": "PaviaU.mat",
        "gt_file": "PaviaU_gt.mat",
        "data_key": "paviaU",
        "gt_key": "paviaU_gt",
        "class_name": ['Asphalt','Meadows','Gravel','Trees','Painted metal sheets',
        'Bare Soil','Bitumen','Self-Blocking Bricks','Shadows']
    },
    "SA": {
        "folder": "SA",
        "description": "Salinas (corrected)",
        "data_file": "Salinas_corrected.mat",
        "gt_file": "Salinas_gt.mat",
        "data_key": "salinas_corrected",
        "gt_key": "salinas_gt",
        "class_name": ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow',
        'Fallow_rough_plow','Fallow_smooth','Stubble','Celery','Grapes_untrained',
        'Soil_vinyard_develop','Corn_senesced_green','Lettuce_romaine_4wk',
        'Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
        'Vinyard_untrained','Vinyard_vertical'],
    },
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def load_hsi_data(dataset: str):
    entry = DATASETS[dataset]

    data_path = DATA_DIR / entry["folder"] / entry["data_file"]
    gt_path = DATA_DIR / entry["folder"] / entry["gt_file"]

    print("Loading:", data_path)
    print("Loading:", gt_path)

    data_mat = loadmat(data_path)
    gt_mat = loadmat(gt_path)

    X = data_mat[entry["data_key"]]
    y = gt_mat[entry["gt_key"]]

    return X, y

def apply_pca(hsi_cube: np.ndarray, n_components: int) -> np.ndarray:
    """
    hsi_cube: (H, W, B)
    returns:  (H, W, n_components)
    """
    H, W, B = hsi_cube.shape

    # Flatten spatial dimensions
    X = hsi_cube.reshape(-1, B)   # (H*W, B)

    # PCA on spectral dimension
    pca = PCA(n_components=n_components, whiten=True)
    X_pca = pca.fit_transform(X)  # (H*W, n_components)

    # Reshape back to image cube
    X_pca = X_pca.reshape(H, W, n_components)

    return X_pca

def pad_with_zeros(X, margin=12):
    h, w, c = X.shape
    out = np.zeros((h + 2*margin, w + 2*margin, c), dtype=X.dtype)
    out[margin:margin+h, margin:margin+w, :] = X
    return out

def create_patches(X, y, patch_size=25, skip_background_patches=False):
    margin = patch_size // 2
    X_pad = pad_with_zeros(X, margin)

    patches = []
    labels = []

    for r in range(y.shape[0]):
        for c in range(y.shape[1]):
            label = y[r, c]
            if skip_background_patches and label == 0:
                continue

            patch = X_pad[r:r+patch_size, c:c+patch_size, :]
            patches.append(patch)
            labels.append(label - 1)   # shift to 0-based classes

    return np.array(patches), np.array(labels)

class HSIDataset(Dataset):
    def __init__(self, X, y):
        # Reshape for HybridSN input
        X_transpose = np.transpose(X, (0, 3, 1, 2))
        self.X = torch.tensor(X_transpose, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HybridSN(nn.Module):
    def __init__(self, num_classes: int, num_bands: int = 30, patch_size: int = 25, dropout: float = 0.4):
        super().__init__()

        self.num_classes = num_classes
        self.num_bands = num_bands
        self.patch_size = patch_size

        # Input: (N, 1, Bands, Patch, Patch)
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3)),
            nn.ReLU(inplace=True),
        )

        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
            nn.ReLU(inplace=True),
        )
        
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True),
        )

        # We infer the 2D conv input size dynamically, so this work for 30 PCA bands
        # and also for 15-band setups
        self.con2d_in_channels, self.flatten_dim = self._infer_shapes(num_bands, patch_size)

        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.con2d_in_channels, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        # x: (N, 1, Bands, Patch, Patch)
        x = self.conv3d_1(x)  # (N, 8, 24, 23, 23) for 30 bands / 25 patch
        x = self.conv3d_2(x)  # (N, 16, 20, 21, 21)
        x = self.conv3d_3(x)  # (N, 32, 18, 19, 19)

        # Merge spectral-depth and channel dims for 2D conv:
        # (N, 32, D, H, W) -> (N, 32*D, H, W)
        n, c, d, h, w = x.shape
        x = x.reshape(n, c * d, h, w)  # (N, 576, 19, 19)
        x = self.conv2d_1(x)  # (N, 64, 17, 17)
        x = x.reshape(x.size(0), -1)  # (N, 18496) for 30 bands / 25x25 setup
        return x

    def _infer_shapes(self, num_bands: int, patch_size: int):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_bands, patch_size, patch_size)
            x = self.conv3d_1(dummy)
            x = self.conv3d_2(x)
            x = self.conv3d_3(x)
            _, c, d, h, w = x.shape
            conv2d_in_channels = c * d

            x = x. reshape(1, conv2d_in_channels, h, w)
            x = nn.Conv2d(conv2d_in_channels, 64, kernel_size=3)(x)
            flatten_dim = x.numel()

        return conv2d_in_channels, flatten_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = self.classifier(x)
        return x

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        out = model(xb)
        loss = criterion(out, yb)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        out = model(xb)
        loss = criterion(out, yb)

        running_loss += loss.item() * xb.size(0)

        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
        #print("true :", yb)
        #print("pred :", preds)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    all_preds = []
    all_true = []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.append(preds)
        all_true.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    return y_true, y_pred

def create_patches_with_positions(X, gt, patch_size, remove_zero=True):
    """
    Create image patches and keep each patch centre coordinate.

    X: PCA-reduced HSI cube, shape (H, W, bands)
    gt: ground-truth map, shape (H, W), where 0 is background
    patch_size: spatial patch size, e.g. 25
    remove_zero: if True, skip background pixels and convert labels to 0-based

    Returns:
        patches:   (N, patch_size, patch_size, bands)
        labels:    (N,), 0-based class labels when remove_zero=True
        positions: (N, 2), original centre pixel coordinates as (row, col)
    """
    margin = patch_size // 2
    H, W = gt.shape

    X_pad = np.pad(
        X,
        ((margin, margin), (margin, margin), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    patches = []
    labels = []
    positions = []

    for r in range(H):
        for c in range(W):
            label = gt[r, c]

            if remove_zero and label == 0:
                continue

            patch = X_pad[r:r + patch_size, c:c + patch_size, :]
            patches.append(patch)
            positions.append((r, c))

            # For labelled HSI data, gt labels usually start at 1.
            # PyTorch CrossEntropyLoss requires labels from 0 to NUM_CLASS-1.
            labels.append(label - 1 if remove_zero else label)

    return (
        np.asarray(patches, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(positions, dtype=np.int64),
    )

def block_split_indices_class_aware(
    positions,
    labels,
    image_shape,
    block_size,
    train_size=0.3,
    val_size=0.1,
    seed=0,
    num_classes=None,
    max_tries=2000,
    without_validation=False,
):
    """
    Class-aware spatial block split.

    positions: array of (row, col) for each patch centre
    labels: zero-based labels for each patch
    image_shape: original ground truth shape, e.g. y.shape
    block_size: spatial block size, e.g. 48 or 64
    train_size: approximate ratio of labelled samples for train
    val_size: approximate ratio of labelled samples for validation

    Returns:
        train_idx, val_idx, test_idx
    """
    rng = np.random.default_rng(seed)

    positions = np.asarray(positions)
    labels = np.asarray(labels).astype(int)

    if num_classes is None:
        num_classes = int(labels.max()) + 1

    H, W = image_shape
    n_block_cols = int(np.ceil(W / block_size))

    # Assign each labelled pixel/patch centre to a spatial block
    block_rows = positions[:, 0] // block_size
    block_cols = positions[:, 1] // block_size
    block_ids = block_rows * n_block_cols + block_cols

    unique_blocks, sample_block_idx = np.unique(block_ids, return_inverse=True)
    n_blocks = len(unique_blocks)

    # Build block x class count matrix
    block_class_counts = np.zeros((n_blocks, num_classes), dtype=int)

    for b_idx, label in zip(sample_block_idx, labels):
        block_class_counts[b_idx, label] += 1

    block_sample_counts = block_class_counts.sum(axis=1)
    block_has_class = block_class_counts > 0

    # Check whether every class appears in at least 3 blocks.
    # Otherwise train/val/test cannot all contain that class.
    blocks_per_class = block_has_class.sum(axis=0)

    impossible_classes = np.where(blocks_per_class < 3)[0]

    if len(impossible_classes) > 0:
        print(
            "Warning: Some classes appear in fewer than 3 spatial blocks, "
            "so train/val/test cannot all contain every class."
        )
        print("Problem class IDs:", impossible_classes)
        RuntimeError("Please reduce block size or use random split instead.")

    
    if(without_validation):
        split_names = ["train", "test"]
        target_ratios = np.array([
            train_size,
            1.0 - train_size,
        ])
    else:
        split_names = ["train", "val", "test"]
        target_ratios = np.array([
            train_size,
            val_size,
            1.0 - train_size - val_size,
        ])

    total_samples = len(labels)
    target_samples = target_ratios * total_samples

    best_assignment = None
    best_score = float("inf")

    for _ in range(max_tries):
        block_assignment = np.full(n_blocks, -1, dtype=int)

        split_class_counts = np.zeros((len(split_names), num_classes), dtype=int)
        split_sample_counts = np.zeros(len(split_names), dtype=int)

        success = True

        # Handle rare classes first
        class_order = np.argsort(blocks_per_class)

        for cls in class_order:
            for split_id in range(len(split_names)):
                if split_class_counts[split_id, cls] > 0:
                    continue

                candidate_blocks = np.where(
                    (block_assignment == -1) &
                    (block_class_counts[:, cls] > 0)
                )[0]

                if len(candidate_blocks) == 0:
                    # Maybe the class is already available in this split
                    if split_class_counts[split_id, cls] == 0:
                        success = False
                    break

                # Prefer blocks rich in this class, but add randomness
                cls_counts = block_class_counts[candidate_blocks, cls]
                top_k = min(5, len(candidate_blocks))
                top_candidates = candidate_blocks[
                    np.argsort(cls_counts)[-top_k:]
                ]

                chosen_block = rng.choice(top_candidates)

                block_assignment[chosen_block] = split_id
                split_class_counts[split_id] += block_class_counts[chosen_block]
                split_sample_counts[split_id] += block_sample_counts[chosen_block]

            if not success:
                break

        if not success:
            continue

        # Assign remaining blocks while trying to match target ratios
        remaining_blocks = np.where(block_assignment == -1)[0]
        rng.shuffle(remaining_blocks)

        for block in remaining_blocks:
            deficits = target_samples - split_sample_counts

            # Put the block into the split most below its target
            split_id = int(np.argmax(deficits))

            block_assignment[block] = split_id
            split_class_counts[split_id] += block_class_counts[block]
            split_sample_counts[split_id] += block_sample_counts[block]

        # Check whether every split has every class
        class_coverage_ok = np.all(split_class_counts > 0)

        # Score based on how close the split sample ratios are to target
        actual_ratios = split_sample_counts / split_sample_counts.sum()
        ratio_error = np.abs(actual_ratios - target_ratios).sum()

        # Penalize missing classes heavily
        missing_class_penalty = np.sum(split_class_counts == 0) * 1000

        score = ratio_error + missing_class_penalty

        if score < best_score:
            best_score = score
            best_assignment = block_assignment.copy()

        if class_coverage_ok:
            break

    if best_assignment is None:
        raise RuntimeError("Could not create a valid block split.")

    if(without_validation):
        train_blocks = np.where(best_assignment == 0)[0]
        test_blocks = np.where(best_assignment == 1)[0]

        train_idx = np.where(np.isin(sample_block_idx, train_blocks))[0]
        test_idx = np.where(np.isin(sample_block_idx, test_blocks))[0]

        block_split_summary = "\nBlock split summary:"
        print("\nBlock split summary:")
        for split_id, name in enumerate(split_names):
            idx = [train_idx, test_idx][split_id]
            split_labels = labels[idx]

            block_split_summary += f"\n{name}: {len(idx)} samples, ratio={100 * len(idx) / len(labels):.2f}"
            print(f"{name}: {len(idx)} samples, ratio={100 * len(idx) / len(labels):.2f}")

            class_counts = np.bincount(split_labels, minlength=num_classes)
            missing = np.where(class_counts == 0)[0]

            if len(missing) > 0:
                print(f"  Missing classes: {missing}")
            else:
                print("  All classes present.")

        return train_idx, None, test_idx, block_split_summary
    else:
        train_blocks = np.where(best_assignment == 0)[0]
        val_blocks = np.where(best_assignment == 1)[0]
        test_blocks = np.where(best_assignment == 2)[0]

        train_idx = np.where(np.isin(sample_block_idx, train_blocks))[0]
        val_idx = np.where(np.isin(sample_block_idx, val_blocks))[0]
        test_idx = np.where(np.isin(sample_block_idx, test_blocks))[0]

        block_split_summary = "\nBlock split summary:"
        print("\nBlock split summary:")
        for split_id, name in enumerate(split_names):
            idx = [train_idx, val_idx, test_idx][split_id]
            split_labels = labels[idx]

            block_split_summary += f"\n{name}: {len(idx)} samples, ratio={100 * len(idx) / len(labels):.2f}"
            print(f"{name}: {len(idx)} samples, ratio={100 * len(idx) / len(labels):.2f}")

            class_counts = np.bincount(split_labels, minlength=num_classes)
            missing = np.where(class_counts == 0)[0]

            if len(missing) > 0:
                print(f"  Missing classes: {missing}")
            else:
                print("  All classes present.")

        return train_idx, val_idx, test_idx, block_split_summary

def get_labelled_positions_and_labels(gt):
    """
    Get labelled pixel coordinates before PCA or patch creation.

    gt uses 0 as background and 1..C as class labels.
    The returned labels are converted to 0..C-1 for PyTorch/sklearn metrics.
    """
    rows, cols = np.nonzero(gt)
    positions = np.stack([rows, cols], axis=1)
    labels = gt[rows, cols].astype(np.int64) - 1
    return positions, labels


def fit_pca_on_train_pixels_and_transform_full_cube(X, train_positions, n_components, seed=0, whiten=True):
    """
    Fit PCA ONLY on training pixel spectra, then transform the full HSI cube.

    This avoids PCA leakage from validation/test pixels while still producing
    a PCA image cube so normal patch extraction can be used afterwards.
    """
    H, W, B = X.shape

    train_rows = train_positions[:, 0]
    train_cols = train_positions[:, 1]
    train_spectra = X[train_rows, train_cols, :].reshape(-1, B)

    pca = PCA(n_components=n_components, whiten=whiten, random_state=seed)
    pca.fit(train_spectra)

    X_flat = X.reshape(-1, B)
    X_pca_flat = pca.transform(X_flat)
    X_pca = X_pca_flat.reshape(H, W, n_components).astype(np.float32)

    return X_pca, pca

def patch_indices_from_split_indices(patch_positions, split_positions, selected_split_indices):
    """
    Convert split indices based on split_positions into indices for X_all/y_all.
    """
    selected_positions = {
        (int(split_positions[i, 0]), int(split_positions[i, 1]))
        for i in selected_split_indices
    }

    patch_indices = [
        i for i, pos in enumerate(patch_positions)
        if (int(pos[0]), int(pos[1])) in selected_positions
    ]

    return np.asarray(patch_indices, dtype=np.int64)