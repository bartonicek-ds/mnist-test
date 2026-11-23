import logging
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn

from config import Config
from model import evaluate_model, make_model
from result import is_err
from utils import download_mnist_if_missing, load_split


def setup_mlflow(tracking_uri: str, experiment: str, artifacts_dir: Path) -> None:
    mlflow.set_tracking_uri(tracking_uri)

    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifacts_dir.as_uri())

    mlflow.set_experiment(experiment)


def main() -> None:
    torch.manual_seed(Config.SEED)
    device = torch.device("cpu")
    logging.info(f"Device: {device} ({torch.get_num_threads()} threads)")

    downloaded = download_mnist_if_missing(Config.MNIST_URL, Config.MNIST_FILES)
    if is_err(downloaded):
        raise downloaded

    train_split = load_split(Config.TRAIN_IMAGES, Config.TRAIN_LABELS)
    if is_err(train_split):
        raise train_split

    test_split = load_split(Config.TEST_IMAGES, Config.TEST_LABELS)
    if is_err(test_split):
        raise test_split

    # The whole dataset fits in memory, so move it to the device up front
    x_train, y_train = (tensor.to(device) for tensor in train_split)
    x_test, y_test = (tensor.to(device) for tensor in test_split)

    setup_mlflow(Config.MLFLOW_TRACKING_URI, Config.MLFLOW_EXPERIMENT, Config.MLFLOW_ARTIFACTS_DIR)
    with mlflow.start_run() as run:
        mlflow.log_params({"epochs": Config.EPOCHS, "batch_size": Config.BATCH_SIZE,
                           "lr": Config.LR, "seed": Config.SEED, "device": str(device)})

        model = make_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, Config.EPOCHS + 1):
            model.train()
            start = time.perf_counter()
            running_loss = 0.0
            permutation = torch.randperm(len(x_train), device=device)  # new random order each epoch

            for batch_start in range(0, len(x_train), Config.BATCH_SIZE):
                batch_indices = permutation[batch_start:batch_start + Config.BATCH_SIZE]
                x_batch, y_batch = x_train[batch_indices], y_train[batch_indices]
                optimizer.zero_grad()

                loss = loss_fn(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * y_batch.size(0)

            elapsed = time.perf_counter() - start
            train_loss = running_loss / len(x_train)
            acc = evaluate_model(model, x_test, y_test)

            logging.info(f"Finished epoch {epoch}/{Config.EPOCHS}  loss {train_loss:.4f}  "
                         f"test acc {acc:.2%}  ({elapsed:.1f}s)")
            mlflow.log_metrics({"train_loss": train_loss, "test_acc": acc, "epoch_seconds": elapsed},
                               step=epoch)

        Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = Config.MODEL_DIR / "model.pt"
        torch.save(model.state_dict(), model_path)
        logging.info(f"Saved model to {model_path}")

        input_example = x_test[:1].cpu().numpy()
        mlflow.pytorch.log_model(model, name="model", input_example=input_example, serialization_format="pickle")
        logging.info(f"Logged MLflow run {run.info.run_id}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    main()
