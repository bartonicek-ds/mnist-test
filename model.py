import torch
from torch import Tensor, nn


def make_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x14x14
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64x7x7
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(128, 10),
    )


def evaluate_model(model: nn.Module, x: Tensor, y: Tensor, batch_size: int = 1000) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_start in range(0, len(x), batch_size):
            batch_end = batch_start + batch_size
            x_batch, y_batch = x[batch_start:batch_end], y[batch_start:batch_end]
            correct += (model(x_batch).argmax(dim=1) == y_batch).sum().item()

    return correct / len(y)
