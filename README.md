# MNIST CNN

Trains a small convolutional neural network (CNN) to classify handwritten digits from the
[MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset, on the CPU, and tracks each run
with [MLflow](https://mlflow.org/).

It reaches about **99% test accuracy in 3 epochs**, taking roughly 12 seconds per epoch on a laptop CPU.

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs the CPU-only build of PyTorch (no CUDA, no GPU needed).
On macOS, change `torch==2.15.0+cpu` to `torch==2.15.0`, since the `+cpu` build only exists
for Linux and Windows.

## Training

```bash
python train.py
```

The script:

1. Downloads the four MNIST files into `data/mnist/` (about 12 MB) from Google's mirror,
   checking each file's MD5 checksum. Files already present and valid are skipped.
2. Loads the whole dataset into memory: 60,000 training and 10,000 test images of 28×28 pixels.
3. Trains the CNN, logging training loss and test accuracy after every epoch.
4. Saves the trained weights to `data/models/model.pt` and logs the run to MLflow.

Example output:

```
INFO Finished epoch 1/3  loss 0.1955  test acc 98.36%  (11.6s)
INFO Finished epoch 2/3  loss 0.0630  test acc 98.85%  (12.2s)
INFO Finished epoch 3/3  loss 0.0448  test acc 99.08%  (12.0s)
```

## Viewing runs in MLflow

Each run records its parameters (epochs, batch size, learning rate, seed), per-epoch metrics
(`train_loss`, `test_acc`, `epoch_seconds`) and the trained model. Runs are stored locally in
`data/mlflow.db` (SQLite), and model files in `data/mlruns/`.

```bash
mlflow ui --backend-store-uri sqlite:///data/mlflow.db
```

Then open http://127.0.0.1:5000.

To load a logged model back:

```python
import mlflow.pytorch
model = mlflow.pytorch.load_model("runs:/<run_id>/model")
```

## Configuration

All settings live in the `Config` class in `config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `EPOCHS` | `3` | Passes over the training set |
| `BATCH_SIZE` | `128` | Images per training step |
| `LR` | `1e-3` | Learning rate for the Adam optimizer |
| `SEED` | `0` | Random seed, for reproducible runs |
| `DATA_DIR` | `data/mnist` | Where the MNIST files are stored |
| `MODEL_DIR` | `data/models` | Where `model.pt` is saved |
| `MLFLOW_TRACKING_URI` | `sqlite:///data/mlflow.db` | Where MLflow stores runs |
| `MLFLOW_EXPERIMENT` | `mnist` | MLflow experiment name |

## Model

```
Input 1×28×28
 → Conv 3×3, 32 filters → ReLU → MaxPool 2×2   (32×14×14)
 → Conv 3×3, 64 filters → ReLU → MaxPool 2×2   (64×7×7)
 → Flatten → Linear 3136→128 → ReLU → Dropout 0.25
 → Linear 128→10                                (one score per digit)
```

About 420,000 parameters, trained with the Adam optimizer and cross-entropy loss.

## Project structure

| File | Contents |
|---|---|
| `train.py` | Entry point: download, load data, train, evaluate, save and log to MLflow |
| `model.py` | `make_model()` (the CNN) and `evaluate_model()` (accuracy on a dataset) |
| `utils.py` | Downloading with checksum verification, and reading the MNIST IDX files |
| `config.py` | All paths, hyperparameters and MLflow settings |
| `requirements.txt` | Python dependencies |
