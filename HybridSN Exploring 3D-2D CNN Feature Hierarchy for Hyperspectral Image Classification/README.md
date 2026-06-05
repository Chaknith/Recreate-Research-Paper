# HybridSN

This repository recreates the HybridSN implementation and runs benchmark experiments under different configurations.

- **[Detailed Summary of the Paper](<Paper's summary.md>)**
- **[Findings Note](<Findings.md>)**

Official GitHub repositories:

- TensorFlow version: <https://github.com/gokriznastic/HybridSN>
- PyTorch version: <https://github.com/Pancakerr/HybridSN>

## Scientific Verification ✅

The recreated implementation achieves results close to the official PyTorch version. This suggests that the HybridSN architecture is robust and that its performance is not merely an artifact of a specific software environment or implementation detail.

For more details, see the [baseline findings](<Findings.md#Baseline>).

## Running the Project

The `recreate.py` script is the closest recreation of the author's implementation. A Jupyter Notebook version is also available as `recreate.ipynb`.

```shell
python3 recreate.py
```

### Downloading the Data

Run the download script before training or benchmarking:

```Shell
python3 download_data.py
```

### Python Scripts

- [download_data.py](script/download_data.py): Downloads the required datasets.
- [helper.py](script/helper.py): Contains shared utility functions used by other scripts.
- [recreate.py](recreate.py): Provides the closest replication of the official implementation.
- [benchmark.py](script/benchmark.py): Runs the benchmark experiment on the unmodified replica.
- [without_valid_benchmark.py](script/without_valid_benchmark.py): Runs the benchmark without a validation set, using only training and test sets.
- [datasplit_benchmark.py](script/datasplit_benchmark.py): Splits the data by spatial blocks instead of individual pixels.
- [pca_control_benchmark.py](script/pca_control_benchmark.py): Fits PCA using only the training data, then applies the transformation to the validation and test sets.
- [calculate_sd.py](script/calculate_sd.py): Calculates the standard deviation of the benchmark results.