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
        "pca": 30,
        "class_name": [ "Alfalfa","Corn-notill","Corn-mintill","Corn","Grass-pasture",
        "Grass-trees","Grass-pasture-mowed","Hay-windrowed","Oats","Soybean-notill",
        "Soybean-mintill","Soybean-clean","Wheat","Woods","Buildings-Grass-Trees-Drives",
        "Stone-Steel-Towers"],
    },
    "UP": {
        "folder": "UP",
        "description": "Pavia University",
        "data_file": "PaviaU.mat",
        "gt_file": "PaviaU_gt.mat",
        "pca": 15,
        "class_name": ['Asphalt','Meadows','Gravel','Trees','Painted metal sheets',
        'Bare Soil','Bitumen','Self-Blocking Bricks','Shadows']
    },
    "SA": {
        "folder": "SA",
        "description": "Salinas (corrected)",
        "data_file": "Salinas_corrected.mat",
        "gt_file": "Salinas_gt.mat",
        "pca": 15,
        "class_name": [ 'Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow',
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

    X = data_mat["indian_pines_corrected"]#lower(entry["data_key"])
    y = gt_mat["indian_pines_gt"]#lower(entry["gt_key"])

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
    pca = PCA(n_components=n_components, whiten=False)
    X_pca = pca.fit_transform(X)  # (H*W, n_components)

    # Reshape back to image cube
    X_pca = X_pca.reshape(H, W, n_components)

    return X_pca