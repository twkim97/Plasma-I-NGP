from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


TRAIN_TEST_SPLIT = {
    10.0: {
        "train": [0.01, 0.03, 0.1, 1.0, 1.5, 2.5],
        "test": [0.05, 0.3, 0.5, 1.3, 2.0, 2.3],
    },
    5.0: {
        "train": [0.01, 0.05, 0.5, 1.3, 2.0, 2.5],
        "test": [0.03, 0.3, 1.0, 1.5, 2.3],
    },
}


class PlasmaDataset(Dataset):
    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
        dtype: torch.dtype = torch.float32,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.dtype = dtype

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        if not self.output_path.exists():
            raise FileNotFoundError(f"Output file not found: {self.output_path}")

        x = np.load(self.input_path)
        y = np.load(self.output_path)

        self._validate_arrays(x, y)

        self.x = torch.as_tensor(x, dtype=self.dtype)
        self.y = torch.as_tensor(y, dtype=self.dtype)

    @staticmethod
    def _validate_arrays(x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 2:
            raise ValueError(f"Input array must be 2D, but got shape {x.shape}")

        if y.ndim != 2:
            raise ValueError(f"Output array must be 2D, but got shape {y.shape}")

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Input and output must have the same number of samples. "
                f"Got {x.shape[0]} and {y.shape[0]}."
            )

        if x.shape[1] != 5:
            raise ValueError(
                f"Input must have shape (N, 5) = [r, z, t, J, tau_ns]. "
                f"Got {x.shape}."
            )

        if y.shape[1] != 3:
            raise ValueError(
                f"Output must have shape (N, 3) = [v, p, rho]. "
                f"Got {y.shape}."
            )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def normalize_energy(J: float) -> float:


    return float(np.log1p(J) / 10.0)


def normalize_tau(tau_ns: float) -> float:


    return float(tau_ns / 100.0)


def build_condition_mask(
    x: np.ndarray,
    split: Literal["train", "test"],
    condition_split: Dict[float, Dict[str, List[float]]] = TRAIN_TEST_SPLIT,
    atol: float = 1.0e-8,
) -> np.ndarray:


    if split not in ["train", "test"]:
        raise ValueError(f"split must be either 'train' or 'test', but got {split}")

    J_norm = x[:, 3]
    tau_norm = x[:, 4]

    mask = np.zeros(x.shape[0], dtype=bool)

    for tau_ns, split_dict in condition_split.items():
        target_tau = normalize_tau(tau_ns)
        tau_mask = np.isclose(tau_norm, target_tau, atol=atol)

        for J in split_dict[split]:
            target_J = normalize_energy(J)
            J_mask = np.isclose(J_norm, target_J, atol=atol)

            mask |= tau_mask & J_mask

    return mask


def get_condition_indices(
    input_path: str | Path,
    split: Literal["train", "test"],
    condition_split: Dict[float, Dict[str, List[float]]] = TRAIN_TEST_SPLIT,
    atol: float = 1.0e-8,
) -> np.ndarray:

    x = np.load(input_path)

    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError(
            f"Input must have shape (N, 5) = [r, z, t, J, tau_ns]. "
            f"Got {x.shape}."
        )

    mask = build_condition_mask(
        x=x,
        split=split,
        condition_split=condition_split,
        atol=atol,
    )

    return np.where(mask)[0]


def create_condition_dataloaders(
    input_path: str | Path,
    output_path: str | Path,
    batch_size: int = 1024,
    num_workers: int = 4,
    pin_memory: bool = True,
    atol: float = 1.0e-8,
) -> Tuple[DataLoader, DataLoader]:


    dataset = PlasmaDataset(
        input_path=input_path,
        output_path=output_path,
    )

    x_np = dataset.x.cpu().numpy()

    train_mask = build_condition_mask(x_np, split="train", atol=atol)
    test_mask = build_condition_mask(x_np, split="test", atol=atol)

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    if len(train_indices) == 0:
        raise RuntimeError(
            "No training samples were found. "
            "Check whether input[:, 3] and input[:, 4] are normalized as "
            "J_norm=log1p(J)/10 and tau_norm=tau_ns/100."
        )

    if len(test_indices) == 0:
        raise RuntimeError(
            "No test samples were found. "
            "Check whether input[:, 3] and input[:, 4] are normalized as "
            "J_norm=log1p(J)/10 and tau_norm=tau_ns/100."
        )

    overlap = np.intersect1d(train_indices, test_indices)
    if len(overlap) > 0:
        raise RuntimeError(
            f"Train and test splits overlap. Number of overlapping samples: {len(overlap)}"
        )

    train_set = Subset(dataset, train_indices.tolist())
    test_set = Subset(dataset, test_indices.tolist())

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def load_numpy_pair(
    input_path: str | Path,
    output_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:


    x = np.load(input_path)
    y = np.load(output_path)

    PlasmaDataset._validate_arrays(x, y)

    return x, y


def summarize_condition_split(
    input_path: str | Path,
    atol: float = 1.0e-8,
) -> Dict[str, int]:


    x = np.load(input_path)

    train_mask = build_condition_mask(x, split="train", atol=atol)
    test_mask = build_condition_mask(x, split="test", atol=atol)

    used_mask = train_mask | test_mask
    unused_mask = ~used_mask

    return {
        "num_total": int(x.shape[0]),
        "num_train": int(train_mask.sum()),
        "num_test": int(test_mask.sum()),
        "num_unused": int(unused_mask.sum()),
    }