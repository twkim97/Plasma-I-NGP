from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import json
import numpy as np


@dataclass
class MinMaxStats:
    min: np.ndarray
    max: np.ndarray
    eps: float = 1.0e-12


@dataclass
class PSLNStats:
    scale: np.ndarray
    min: np.ndarray
    max: np.ndarray
    percentile: float
    scale_constant: float
    eps: float = 1.0e-12


def signed_log_transform(x: np.ndarray, scale: np.ndarray) -> np.ndarray:


    return np.sign(x) * np.log1p(np.abs(x) / scale)


def signed_log_inverse(z: np.ndarray, scale: np.ndarray) -> np.ndarray:


    return np.sign(z) * scale * np.expm1(np.abs(z))


def compute_minmax_stats(x: np.ndarray, eps: float = 1.0e-12) -> MinMaxStats:


    return MinMaxStats(
        min=np.min(x, axis=0),
        max=np.max(x, axis=0),
        eps=eps,
    )


def minmax_normalize(x: np.ndarray, stats: MinMaxStats) -> np.ndarray:


    denom = np.maximum(stats.max - stats.min, stats.eps)
    return (x - stats.min) / denom


def minmax_inverse(x_norm: np.ndarray, stats: MinMaxStats) -> np.ndarray:


    return x_norm * (stats.max - stats.min) + stats.min


def compute_psln_stats(
    y: np.ndarray,
    percentile: float = 99.0,
    scale_constant: float = 1.0e9,
    eps: float = 1.0e-12,
) -> PSLNStats:


    abs_percentile = np.percentile(np.abs(y), percentile, axis=0)
    scale = np.maximum(abs_percentile / scale_constant, eps)

    y_log = signed_log_transform(y, scale)
    y_min = np.min(y_log, axis=0)
    y_max = np.max(y_log, axis=0)

    return PSLNStats(
        scale=scale,
        min=y_min,
        max=y_max,
        percentile=percentile,
        scale_constant=scale_constant,
        eps=eps,
    )


def psln_normalize(y: np.ndarray, stats: PSLNStats) -> np.ndarray:


    y_log = signed_log_transform(y, stats.scale)
    denom = np.maximum(stats.max - stats.min, stats.eps)

    return (y_log - stats.min) / denom


def psln_inverse(y_norm: np.ndarray, stats: PSLNStats) -> np.ndarray:
    """
    Inverse PSLN normalization.
    """

    y_log = y_norm * (stats.max - stats.min) + stats.min
    
    return signed_log_inverse(y_log, stats.scale)




def normalize_input_coordinates(
    x: np.ndarray
) -> np.ndarray:


    r = x[:, 0]
    z = x[:, 1]
    t = x[:, 2] / 100
    J = np.log1p(x[:, 3])/10
    tau = x[:, 4] / 100
    x_norm = np.stack((r, z, t, J, tau), axis=1)

    stats = compute_minmax_stats(x)
    return x_norm, stats
    

def make_coordinate_grid(
    r_values: np.ndarray,
    z_values: np.ndarray,
    t_value: float,
    J_value: float,
    tau_ns_value: float,
) -> np.ndarray:

    rr, zz = np.meshgrid(r_values, z_values, indexing="ij")

    n_points = rr.size

    t = np.full((n_points, 1), t_value, dtype=np.float32)
    J = np.full((n_points, 1), J_value, dtype=np.float32)
    tau = np.full((n_points, 1), tau_ns_value, dtype=np.float32)

    x = np.concatenate(
        [
            rr.reshape(-1, 1),
            zz.reshape(-1, 1),
            t,
            J,
            tau,
        ],
        axis=1,
    ).astype(np.float32)

    return x


def flatten_fields(
    velocity: np.ndarray,
    pressure: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:


    if not (velocity.shape == pressure.shape == density.shape):
        raise ValueError(
            "velocity, pressure, and density must have the same shape. "
            f"Got {velocity.shape}, {pressure.shape}, {density.shape}."
        )

    y = np.stack(
        [
            velocity.reshape(-1),
            pressure.reshape(-1),
            density.reshape(-1),
        ],
        axis=1,
    ).astype(np.float32)

    return y


def save_preprocessing_stats(
    path: str | Path,
    input_stats: MinMaxStats,
    output_stats: PSLNStats,
) -> None:


    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    stats_dict = {
        "input_stats": {
            "min": input_stats.min.tolist(),
            "max": input_stats.max.tolist(),
            "eps": input_stats.eps,
        },
        "output_stats": {
            "scale": output_stats.scale.tolist(),
            "min": output_stats.min.tolist(),
            "max": output_stats.max.tolist(),
            "percentile": output_stats.percentile,
            "scale_constant": output_stats.scale_constant,
            "eps": output_stats.eps,
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)


def load_preprocessing_stats(path: str | Path) -> Tuple[MinMaxStats, PSLNStats]:

    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        stats_dict = json.load(f)

    input_stats = MinMaxStats(
        min=np.array(stats_dict["input_stats"]["min"], dtype=np.float32),
        max=np.array(stats_dict["input_stats"]["max"], dtype=np.float32),
        eps=stats_dict["input_stats"].get("eps", 1.0e-12),
    )

    output_stats = PSLNStats(
        scale=np.array(stats_dict["output_stats"]["scale"], dtype=np.float32),
        min=np.array(stats_dict["output_stats"]["min"], dtype=np.float32),
        max=np.array(stats_dict["output_stats"]["max"], dtype=np.float32),
        percentile=stats_dict["output_stats"]["percentile"],
        scale_constant=stats_dict["output_stats"]["scale_constant"],
        eps=stats_dict["output_stats"].get("eps", 1.0e-12),
    )

    return input_stats, output_stats


def preprocess_and_save(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    output_dir: str | Path,
    percentile: float = 99.0,
    scale_constant: float = 1.0e9,
) -> None:


    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_norm, input_stats = normalize_input_coordinates(x_raw)

    output_stats = compute_psln_stats(
        y_raw,
        percentile=percentile,
        scale_constant=scale_constant,
    )
    y_norm = psln_normalize(y_raw, output_stats)

    np.save(output_dir / "sample_input.npy", x_norm.astype(np.float32))
    np.save(output_dir / "sample_output.npy", y_norm.astype(np.float32))

    save_preprocessing_stats(
        output_dir / "preprocessing_stats.json",
        input_stats=input_stats,
        output_stats=output_stats,
    )