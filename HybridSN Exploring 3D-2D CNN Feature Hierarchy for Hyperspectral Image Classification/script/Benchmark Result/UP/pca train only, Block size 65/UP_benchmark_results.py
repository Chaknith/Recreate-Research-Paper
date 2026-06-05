# Auto-generated benchmark results
# Configuration
# seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# DATASET = 'UP'      # IP, UP, SA
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
# BLOCK_SIZE = 65
# SPLIT_TYPE = 'spatial_block_split_train_only_pca'
# PCA_FIT = 'training labelled pixels only'

results = [{'seed': 1, 'block_split_summary': '\nBlock split summary:\ntrain: 10923 samples, ratio=25.54\nval: 7892 samples, ratio=18.45\ntest: 23961 samples, ratio=56.02', 'pca_fit_samples': 10923, 'pca_explained_variance_ratio_sum': 0.9988629043830846, 'OA': 0.7270982012436876, 'AA': 0.779594888693296, 'Kappa': 0.6686358291624158}, {'seed': 2, 'block_split_summary': '\nBlock split summary:\ntrain: 11360 samples, ratio=26.56\nval: 7228 samples, ratio=16.90\ntest: 24188 samples, ratio=56.55', 'pca_fit_samples': 11360, 'pca_explained_variance_ratio_sum': 0.9991844054868063, 'OA': 0.7876219613031256, 'AA': 0.8443933125502148, 'Kappa': 0.7233796553139973}, {'seed': 3, 'block_split_summary': '\nBlock split summary:\ntrain: 12091 samples, ratio=28.27\nval: 5418 samples, ratio=12.67\ntest: 25267 samples, ratio=59.07', 'pca_fit_samples': 12091, 'pca_explained_variance_ratio_sum': 0.9988301486344615, 'OA': 0.7387105711006451, 'AA': 0.7830740287582993, 'Kappa': 0.6733771469794374}, {'seed': 4, 'block_split_summary': '\nBlock split summary:\ntrain: 12930 samples, ratio=30.23\nval: 5084 samples, ratio=11.89\ntest: 24762 samples, ratio=57.89', 'pca_fit_samples': 12930, 'pca_explained_variance_ratio_sum': 0.999377500235488, 'OA': 0.8270333575640094, 'AA': 0.8108715669559375, 'Kappa': 0.7772280578467163}, {'seed': 5, 'block_split_summary': '\nBlock split summary:\ntrain: 11678 samples, ratio=27.30\nval: 4681 samples, ratio=10.94\ntest: 26417 samples, ratio=61.76', 'pca_fit_samples': 11678, 'pca_explained_variance_ratio_sum': 0.9992086064203844, 'OA': 0.8681909376537835, 'AA': 0.813749644902462, 'Kappa': 0.8105263612797329}, {'seed': 6, 'block_split_summary': '\nBlock split summary:\ntrain: 11729 samples, ratio=27.42\nval: 6118 samples, ratio=14.30\ntest: 24929 samples, ratio=58.28', 'pca_fit_samples': 11729, 'pca_explained_variance_ratio_sum': 0.9994264224644168, 'OA': 0.7448754462673994, 'AA': 0.8639066647912402, 'Kappa': 0.6835528657113723}, {'seed': 7, 'block_split_summary': '\nBlock split summary:\ntrain: 12925 samples, ratio=30.22\nval: 4082 samples, ratio=9.54\ntest: 25769 samples, ratio=60.24', 'pca_fit_samples': 12925, 'pca_explained_variance_ratio_sum': 0.9990819397130553, 'OA': 0.8608793511583686, 'AA': 0.8101171267205909, 'Kappa': 0.8117119937621553}, {'seed': 8, 'block_split_summary': '\nBlock split summary:\ntrain: 13707 samples, ratio=32.04\nval: 3974 samples, ratio=9.29\ntest: 25095 samples, ratio=58.67', 'pca_fit_samples': 13707, 'pca_explained_variance_ratio_sum': 0.9990473593641183, 'OA': 0.7091850966327954, 'AA': 0.8172320039734227, 'Kappa': 0.6441282776968309}, {'seed': 9, 'block_split_summary': '\nBlock split summary:\ntrain: 12695 samples, ratio=29.68\nval: 4282 samples, ratio=10.01\ntest: 25799 samples, ratio=60.31', 'pca_fit_samples': 12695, 'pca_explained_variance_ratio_sum': 0.9989630970884433, 'OA': 0.9096476607620451, 'AA': 0.899603183070904, 'Kappa': 0.8787321232934198}, {'seed': 10, 'block_split_summary': '\nBlock split summary:\ntrain: 11441 samples, ratio=26.75\nval: 5651 samples, ratio=13.21\ntest: 25684 samples, ratio=60.04', 'pca_fit_samples': 11441, 'pca_explained_variance_ratio_sum': 0.9994272493733672, 'OA': 0.9052328297772932, 'AA': 0.9159874593563466, 'Kappa': 0.837811196672687}]
