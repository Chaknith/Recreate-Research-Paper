import numpy as np
from pathlib import Path
from typing import Dict
from scipy.io import loadmat
from sklearn.decomposition import PCA

DATASETS: Dict[str, Dict[str, object]] = {
    "IP": {
        "folder": "IP",
        "description": "Indian Pines (corrected)",
        "data_file": "Indian_pines_corrected.mat",
        "gt_file": "Indian_pines_gt.mat",
        "data_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
        "pca": 15,
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
        "pca": 15,
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
        "pca": 15,
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