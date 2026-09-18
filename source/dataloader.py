import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BallMovingDataset(Dataset):
    def __init__(self, base_address):
        self.base_address = Path(base_address)
        if not self.base_address.is_dir():
            raise NotADirectoryError(
                f"Dataset directory not found: {self.base_address}"
            )

        # Keep the actual paths instead of assuming numeric folder names are
        # contiguous (for example, 0, 1, 2, ...).
        self.item_paths = sorted(
            (
                path
                for path in self.base_address.iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            key=lambda path: int(path.name),
        )

    def __getitem__(self, idx):
        item_path = self.item_paths[idx]
        try:
            actions = np.load(item_path / "actions.npy")
            frames = np.load(item_path / "frames.npy")
            with (item_path / "metadata.json").open("r", encoding="utf-8") as file:
                instruction = json.load(file)["case"]["instruction"]
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Missing dataset file in {item_path}") from error

        # actions, frames, instruction, dataset index
        return torch.from_numpy(actions), torch.from_numpy(frames), instruction, idx

    def __len__(self):
        return len(self.item_paths)


class ProcessedVideoDataset(Dataset):
    """Load preprocessing-complete video clips stored as NPZ frame arrays."""

    def __init__(
        self,
        processed_root,
        *,
        split: str,
        expected_shape: tuple[int, int, int, int] = (32, 32, 32, 3),
        array_key: str = "frames",
    ):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.split_root = Path(processed_root) / split
        self.expected_shape = expected_shape
        self.array_key = array_key
        if not self.split_root.is_dir():
            raise FileNotFoundError(f"Processed split not found: {self.split_root}")
        self.clip_paths = sorted(self.split_root.glob("*.npz"))
        if not self.clip_paths:
            raise FileNotFoundError(f"No processed .npz clips in {self.split_root}")

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, index):
        path = self.clip_paths[index]
        with np.load(path, allow_pickle=False) as sample:
            frames = sample[self.array_key]
        if frames.shape != self.expected_shape or frames.dtype != np.uint8:
            raise ValueError(
                f"{path} must contain uint8 {self.array_key} with shape "
                f"{self.expected_shape}; received {frames.dtype} {frames.shape}"
            )
        return torch.from_numpy(frames.copy())
