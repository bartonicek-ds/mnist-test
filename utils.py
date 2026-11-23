import gzip
import hashlib
import logging
import shutil
import struct
import urllib.request
from pathlib import Path

import torch
from torch import Tensor

from result import Result, is_err


class ChecksumError(Exception):
    pass


class IdxFormatError(Exception):
    pass


def md5sum(path: Path) -> Result[str, OSError]:
    digest = hashlib.md5()
    try:
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        return error

    return digest.hexdigest()


def download_file(url: str, dest: Path, md5: str, timeout: float = 60) -> Result[None, OSError | ChecksumError]:
    """Download url to dest, verifying its MD5. Skips the download if dest is already valid."""
    if dest.is_file() and md5sum(dest) == md5:
        logging.info(f"Skipping: {dest.name} already downloaded")
        return None

    tmp = dest.with_name(dest.name + ".part")  # so a failed download never leaves a broken dest
    logging.info(f"Downloading {url}")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as file:
            shutil.copyfileobj(response, file)

        actual = md5sum(tmp)
        if is_err(actual):
            return actual

        if actual != md5:
            return ChecksumError(f"Checksum mismatch for {url}: got {actual}, expected {md5}")

        tmp.replace(dest)
    except OSError as error:
        return error
    finally:
        tmp.unlink(missing_ok=True)

    return None


def download_mnist_if_missing(base_url: str, files: dict[Path, str]) -> Result[None, OSError | ChecksumError]:
    """Download each MNIST file (path -> expected MD5) from base_url, skipping valid ones."""
    for path, md5 in files.items():
        downloaded = download_file(base_url + path.name, path, md5)
        if is_err(downloaded):
            return downloaded

    return None


def load_images(path: Path) -> Result[Tensor, OSError | IdxFormatError]:
    try:
        with gzip.open(path, "rb") as file:
            magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                return IdxFormatError(f"{path} is not an IDX image file (magic number {magic}, expected 2051)")

            data = torch.frombuffer(bytearray(file.read()), dtype=torch.uint8)
    except OSError as error:
        return error
    except struct.error:
        return IdxFormatError(f"{path} has a truncated IDX header")

    # Shape (N, 1, 28, 28), scaled to [0, 1] and normalised with the MNIST mean/std
    images = data.view(count, 1, rows, cols).float() / 255.0
    return (images - 0.1307) / 0.3081


def load_labels(path: Path) -> Result[Tensor, OSError | IdxFormatError]:
    try:
        with gzip.open(path, "rb") as file:
            magic, _count = struct.unpack(">II", file.read(8))
            if magic != 2049:
                return IdxFormatError(f"{path} is not an IDX label file (magic number {magic}, expected 2049)")

            return torch.frombuffer(bytearray(file.read()), dtype=torch.uint8).long()
    except OSError as error:
        return error
    except struct.error:
        return IdxFormatError(f"{path} has a truncated IDX header")


def load_split(images_path: Path, labels_path: Path) -> Result[tuple[Tensor, Tensor], OSError | IdxFormatError]:
    images = load_images(images_path)
    if is_err(images):
        return images

    labels = load_labels(labels_path)
    if is_err(labels):
        return labels

    return images, labels
