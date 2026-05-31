
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from src.dataset import PlasmaDataset
from src.metrics import (
    FIELD_NAMES,
    fieldwise_rmse_np,
    global_rmse_np,
    legacy_condition_score,
)
from src.model import build_model


DEFAULT_TEST_CONDITIONS = {
    10.0: [0.05, 0.3, 0.5, 1.3, 2.0, 2.3],
    5.0: [0.03, 0.3, 1.0, 1.5, 2.3],
}


DEFAULT_TIME_VALUES = list(range(2, 31, 2))


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_time(t_ns: float) -> float:


    return float(t_ns / 100.0)


def normalize_energy(J: float) -> float:


    return float(np.log1p(J) / 10.0)


def normalize_tau(tau_ns: float) -> float:


    return float(tau_ns / 100.0)


def build_condition_time_mask(
    x: np.ndarray,
    tau_ns: float,
    J: float,
    t_ns: float,
    atol: float = 1.0e-8,
) -> np.ndarray:


    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError(
            f"x must have shape (N, 5) = [r, z, t, J, tau_ns]. "
            f"Got {x.shape}."
        )

    target_t = normalize_time(t_ns)
    target_J = normalize_energy(J)
    target_tau = normalize_tau(tau_ns)

    return (
        np.isclose(x[:, 2], target_t, atol=atol)
        & np.isclose(x[:, 3], target_J, atol=atol)
        & np.isclose(x[:, 4], target_tau, atol=atol)
    )


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


@torch.no_grad()
def predict_in_batches(
    model: torch.nn.Module,
    x: np.ndarray | torch.Tensor,
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:


    model.eval()

    if isinstance(x, np.ndarray):
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
    else:
        x_tensor = x.detach().cpu().to(dtype=torch.float32)

    preds = []

    for start in range(0, x_tensor.shape[0], batch_size):
        end = start + batch_size
        x_batch = x_tensor[start:end].to(device, non_blocking=True)

        y_batch = model(x_batch)
        preds.append(y_batch.detach().cpu())

    return torch.cat(preds, dim=0).numpy()
    
def get_model_config(config, model_key: str):
    if model_key == "main":
        return config["model"]

    if "baselines" not in config:
        raise KeyError("No 'baselines' section found in config.")

    if model_key not in config["baselines"]:
        available = list(config["baselines"].keys())
        raise KeyError(f"Unknown model_key: {model_key}. Available: {available}")

    return config["baselines"][model_key]
    
@torch.no_grad()
def evaluate_one_condition_legacy(
    model: torch.nn.Module,
    x_all: np.ndarray,
    y_all: np.ndarray,
    tau_ns: float,
    J: float,
    time_values: List[int] | List[float],
    device: torch.device,
    batch_size: int = 65536,
    atol: float = 1.0e-8,
) -> Dict[str, Any]:


    timewise_field_rmse = {name: [] for name in FIELD_NAMES}
    num_samples_by_time = {}
    missing_times = []

    # Optional auxiliary global arrays for this condition.
    all_preds = []
    all_targets = []

    for t_ns in time_values:
        mask = build_condition_time_mask(
            x=x_all,
            tau_ns=tau_ns,
            J=J,
            t_ns=t_ns,
            atol=atol,
        )

        indices = np.where(mask)[0]

        if len(indices) == 0:
            missing_times.append(float(t_ns))
            continue

        x_t = x_all[indices]
        y_t = y_all[indices]

        pred_t = predict_in_batches(
            model=model,
            x=x_t,
            device=device,
            batch_size=batch_size,
        )

        field_rmse = fieldwise_rmse_np(
            pred=pred_t,
            target=y_t,
            field_names=FIELD_NAMES,
        )

        for name in FIELD_NAMES:
            timewise_field_rmse[name].append(field_rmse[f"rmse_{name}"])

        num_samples_by_time[str(t_ns)] = int(len(indices))

        all_preds.append(pred_t)
        all_targets.append(y_t)

    score = legacy_condition_score(
        timewise_field_rmse=timewise_field_rmse,
        field_names=FIELD_NAMES,
    )

    result = {
        "tau_ns": float(tau_ns),
        "J": float(J),
        "num_time_steps": int(
            min(len(timewise_field_rmse[name]) for name in FIELD_NAMES)
        ),
        "num_samples": int(sum(num_samples_by_time.values())),
        "missing_times": missing_times,
        **score,
    }

    # Auxiliary global RMSE, not used as the main paper metric.
    if len(all_preds) > 0:
        pred_cond = np.concatenate(all_preds, axis=0)
        target_cond = np.concatenate(all_targets, axis=0)
        result["global_RMSE_aux"] = global_rmse_np(
            pred=pred_cond,
            target=target_cond,
        )
    else:
        result["error"] = "No samples found for this condition."

    return result


@torch.no_grad()
def evaluate_by_condition_legacy(
    model: torch.nn.Module,
    dataset: PlasmaDataset,
    device: torch.device,
    test_conditions: Dict[float, List[float]],
    time_values: List[int] | List[float],
    batch_size: int = 65536,
    atol: float = 1.0e-8,
) -> Dict[str, Any]:


    x_all = dataset.x.detach().cpu().numpy()
    y_all = dataset.y.detach().cpu().numpy()

    results = {
        "metric": "SLRMSE",
        "aggregation": "field-wise RMSE at each time step -> RMS over time -> mean over fields",
        "field_names": FIELD_NAMES,
        "time_values": list(time_values),
        "conditions": [],
    }

    for tau_ns, J_list in test_conditions.items():
        for J in J_list:
            condition_result = evaluate_one_condition_legacy(
                model=model,
                x_all=x_all,
                y_all=y_all,
                tau_ns=float(tau_ns),
                J=float(J),
                time_values=time_values,
                device=device,
                batch_size=batch_size,
                atol=atol,
            )

            results["conditions"].append(condition_result)

    valid_scores = [
        item["SLRMSE"]
        for item in results["conditions"]
        if "SLRMSE" in item and not np.isnan(item["SLRMSE"])
    ]

    if len(valid_scores) > 0:
        results["overall_condition_mean"] = {
            "SLRMSE": float(np.mean(valid_scores)),
            "num_conditions": int(len(valid_scores)),
        }

    return results


def get_test_conditions_from_config(config: Dict[str, Any]) -> Dict[float, List[float]]:


    if "split" not in config or "test" not in config["split"]:
        return DEFAULT_TEST_CONDITIONS

    test_cfg = config["split"]["test"]

    return {
        10.0: test_cfg.get("tau_10ns", DEFAULT_TEST_CONDITIONS[10.0]),
        5.0: test_cfg.get("tau_5ns", DEFAULT_TEST_CONDITIONS[5.0]),
    }


def get_time_values_from_config(config: Dict[str, Any]) -> List[int]:


    if "evaluation" not in config:
        return DEFAULT_TIME_VALUES

    eval_cfg = config["evaluation"]

    if "time_values" in eval_cfg:
        return list(eval_cfg["time_values"])

    t_start = eval_cfg.get("t_start", 2)
    t_end = eval_cfg.get("t_end", 30)
    t_step = eval_cfg.get("t_step", 2)

    return list(range(t_start, t_end + 1, t_step))


def save_results_json(results: Dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def save_results_csv(results: Dict[str, Any], output_path: str | Path) -> None:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = results.get("conditions", [])

    if len(rows) == 0:
        return

    fieldnames = [
        "tau_ns",
        "J",
        "num_time_steps",
        "num_samples",
        "SLRMSE",
        "SLRMSE_v",
        "SLRMSE_p",
        "SLRMSE_rho",
        "global_RMSE_aux",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(key, "")
                    for key in fieldnames
                }
            )


def print_results(results: Dict[str, Any]) -> None:
    print("Legacy-style condition-wise evaluation results")
    print(f"Metric: {results['metric']}")
    print(f"Aggregation: {results['aggregation']}")
    print()

    for item in results["conditions"]:
        if "error" in item:
            print(
                f"tau={item['tau_ns']} ns | "
                f"J={item['J']} J | "
                f"{item['error']}"
            )
            continue

        print(
            f"tau={item['tau_ns']:>4.1f} ns | "
            f"J={item['J']:<4} J | "
            f"N={item['num_samples']:<8} | "
            f"T={item['num_time_steps']:<2} | "
            f"SLRMSE={item['SLRMSE']:.6f} | "
            f"v={item['SLRMSE_v']:.6f} | "
            f"p={item['SLRMSE_p']:.6f} | "
            f"rho={item['SLRMSE_rho']:.6f}"
        )

    if "overall_condition_mean" in results:
        print()
        print(
            "Overall condition mean SLRMSE: "
            f"{results['overall_condition_mean']['SLRMSE']:.6f} "
            f"over {results['overall_condition_mean']['num_conditions']} conditions"
        )


def run_evaluation(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    model_key,
    csv_output_path: str | Path | None = None,
    batch_size: int = 65536,
    atol: float = 1.0e-8,
    
) -> None:
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = get_model_config(config, model_key)
    model = build_model(model_config)
    model = model.to(device)
    model = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    dataset = PlasmaDataset(
        input_path=config["data"]["input_path"],
        output_path=config["data"]["output_path"],
    )

    test_conditions = get_test_conditions_from_config(config)
    time_values = get_time_values_from_config(config)

    results = evaluate_by_condition_legacy(
        model=model,
        dataset=dataset,
        device=device,
        test_conditions=test_conditions,
        time_values=time_values,
        batch_size=batch_size,
        atol=atol,
    )

    save_results_json(results, output_path)

    if csv_output_path is not None:
        save_results_csv(results, csv_output_path)

    print_results(results)
    print(f"\nSaved JSON results to: {output_path}")

    if csv_output_path is not None:
        print(f"Saved CSV results to: {csv_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/evaluation_by_condition.json")
    parser.add_argument("--csv-output", type=str, default="outputs/evaluation_by_condition.csv")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument(
    "--model-key",
    type=str,
    default="main",
    help="Model key to evaluate: main, siren, nerf, deeponet",
)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        csv_output_path=args.csv_output,
        batch_size=args.batch_size,
        atol=args.atol,
        model_key=args.model_key,
    )