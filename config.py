from pathlib import Path


class Config:
    # Paths (everything generated lives under data/, which is gitignored)
    ROOT_DIR = Path(__file__).parent / "data"
    DATA_DIR = ROOT_DIR / "mnist"
    MODEL_DIR = ROOT_DIR / "models"
    TRAIN_IMAGES = DATA_DIR / "train-images-idx3-ubyte.gz"
    TRAIN_LABELS = DATA_DIR / "train-labels-idx1-ubyte.gz"
    TEST_IMAGES = DATA_DIR / "t10k-images-idx3-ubyte.gz"
    TEST_LABELS = DATA_DIR / "t10k-labels-idx1-ubyte.gz"

    # MNIST files download url and their md5 checksums
    MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    MNIST_FILES = {
        TRAIN_IMAGES: "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        TRAIN_LABELS: "d53e105ee54ea40749a09fcbcd1e9432",
        TEST_IMAGES: "9fb629c4189551a2d022fa330f9573f3",
        TEST_LABELS: "ec29112dd5afa0611ce80d1b7f02629c",
    }

    # MLflow
    MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
    MLFLOW_ARTIFACTS_DIR = ROOT_DIR / "mlruns"
    MLFLOW_EXPERIMENT = "mnist"

    # Model
    SEED = 0
    LR = 1e-3
    EPOCHS = 3
    BATCH_SIZE = 128
