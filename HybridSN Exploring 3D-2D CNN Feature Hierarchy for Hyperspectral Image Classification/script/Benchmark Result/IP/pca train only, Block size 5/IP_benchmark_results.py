# Auto-generated benchmark results
# Configuration
# seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# DATASET = 'IP'      # IP, UP, SA
# TRAIN_SIZE = 0.3    # approximate ratio of labelled pixels for training blocks
# VAL_SIZE = 0.1      # approximate ratio of labelled pixels for validation blocks
# EPOCH = 100         # number of epochs
# VAL_EPOCH = 5       # interval of validation
# LR = 0.001             # learning rate
# WEIGHT_DECAY = 1e-06
# BATCH_SIZE = 256
# DEVICE = 0          # -1: CPU, 0: cuda:0
# N_PCA = 15
# PATCH_SIZE = 25
# DROP_OUT = 0.4
# BLOCK_SIZE = 5
# SPLIT_TYPE = 'spatial_block_split_train_only_pca'
# PCA_FIT = 'training labelled pixels only'

results = [{'seed': 1, 'block_split_summary': '\nBlock split summary:\ntrain: 3081 samples, ratio=30.06\nval: 1018 samples, ratio=9.93\ntest: 6150 samples, ratio=60.01', 'pca_fit_samples': 3081, 'pca_explained_variance_ratio_sum': 0.9863890252395464, 'OA': 0.8603252032520325, 'AA': 0.7269465801283934, 'Kappa': 0.8401744812965555}, {'seed': 2, 'block_split_summary': '\nBlock split summary:\ntrain: 3085 samples, ratio=30.10\nval: 1024 samples, ratio=9.99\ntest: 6140 samples, ratio=59.91', 'pca_fit_samples': 3085, 'pca_explained_variance_ratio_sum': 0.982698846975445, 'OA': 0.8335504885993485, 'AA': 0.8126687686846851, 'Kappa': 0.8111479096296335}, {'seed': 3, 'block_split_summary': '\nBlock split summary:\ntrain: 3077 samples, ratio=30.02\nval: 1028 samples, ratio=10.03\ntest: 6144 samples, ratio=59.95', 'pca_fit_samples': 3077, 'pca_explained_variance_ratio_sum': 0.983803637589384, 'OA': 0.8707682291666666, 'AA': 0.827860906866211, 'Kappa': 0.8503455436168825}, {'seed': 4, 'block_split_summary': '\nBlock split summary:\ntrain: 3064 samples, ratio=29.90\nval: 1027 samples, ratio=10.02\ntest: 6158 samples, ratio=60.08', 'pca_fit_samples': 3064, 'pca_explained_variance_ratio_sum': 0.9833357801691373, 'OA': 0.8670022734654108, 'AA': 0.8270359332245922, 'Kappa': 0.8473896680903614}, {'seed': 5, 'block_split_summary': '\nBlock split summary:\ntrain: 3066 samples, ratio=29.92\nval: 1032 samples, ratio=10.07\ntest: 6151 samples, ratio=60.02', 'pca_fit_samples': 3066, 'pca_explained_variance_ratio_sum': 0.9828078582014383, 'OA': 0.8557958055600715, 'AA': 0.8310848793463459, 'Kappa': 0.8341691031466107}, {'seed': 6, 'block_split_summary': '\nBlock split summary:\ntrain: 3071 samples, ratio=29.96\nval: 1033 samples, ratio=10.08\ntest: 6145 samples, ratio=59.96', 'pca_fit_samples': 3071, 'pca_explained_variance_ratio_sum': 0.9850581277953487, 'OA': 0.8288039056143206, 'AA': 0.7632342133701322, 'Kappa': 0.8062733192797327}, {'seed': 7, 'block_split_summary': '\nBlock split summary:\ntrain: 3072 samples, ratio=29.97\nval: 1020 samples, ratio=9.95\ntest: 6157 samples, ratio=60.07', 'pca_fit_samples': 3072, 'pca_explained_variance_ratio_sum': 0.9848367584909022, 'OA': 0.8198798115965568, 'AA': 0.8073668945160429, 'Kappa': 0.7948861037424766}, {'seed': 8, 'block_split_summary': '\nBlock split summary:\ntrain: 3070 samples, ratio=29.95\nval: 1029 samples, ratio=10.04\ntest: 6150 samples, ratio=60.01', 'pca_fit_samples': 3070, 'pca_explained_variance_ratio_sum': 0.9849414891128292, 'OA': 0.8008130081300813, 'AA': 0.8030315295802947, 'Kappa': 0.7691234081237118}, {'seed': 9, 'block_split_summary': '\nBlock split summary:\ntrain: 3068 samples, ratio=29.93\nval: 1025 samples, ratio=10.00\ntest: 6156 samples, ratio=60.06', 'pca_fit_samples': 3068, 'pca_explained_variance_ratio_sum': 0.985198078925177, 'OA': 0.8487654320987654, 'AA': 0.7696927680891023, 'Kappa': 0.825242324121006}, {'seed': 10, 'block_split_summary': '\nBlock split summary:\ntrain: 3072 samples, ratio=29.97\nval: 1024 samples, ratio=9.99\ntest: 6153 samples, ratio=60.04', 'pca_fit_samples': 3072, 'pca_explained_variance_ratio_sum': 0.9843306019796346, 'OA': 0.8581179912237933, 'AA': 0.8383198486690371, 'Kappa': 0.8382250198274579}]
