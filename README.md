# Neural Surrogate Model for Laser-Induced Plasma Dynamics

This repository provides the official implementation for a neural surrogate model for predicting laser-induced plasma dynamics.

The model learns continuous spatiotemporal plasma fields conditioned on laser operating parameters. Given normalized coordinates and laser conditions,

```text
x = [r, z, t, J, tau_ns]
```

the model predicts normalized plasma quantities,

```text
y = [v, p, rho]
```

where `v`, `p`, and `rho` denote velocity, pressure, and density, respectively.

---

## Repository Structure

```text
repository/
  README.md
  requirements.txt

  configs/
    main_experiment.yaml

  src/
    model.py
    dataset.py
    preprocessing.py
    train.py
    evaluate.py
    metrics.py

  examples/
    sample_input.npy
    sample_output.npy
    preprocessing_stats.json

  checkpoints/
    README.md
    PINGP_1_final.pt
    PINGP_2_final.pt
    PINGP_3_final.pt
    SIREN_1_final.pt
    SIREN_2_final.pt
    SIREN_3_final.pt
    NERF_1_final.pt
    NERF_2_final.pt
    NERF_3_final.pt
    DON_1_final.pt
    DON_2_final.pt
    DON_3_final.pt

  outputs/
    eval_pingp_seed0.csv
    eval_pingp_seed1.csv
    eval_pingp_seed2.csv
    eval_siren_seed0.csv
    ...
```

---

## Installation

Install the basic dependencies:

```bash
pip install -r requirements.txt
```

The proposed Instant-NGP-based model requires `tiny-cuda-nn`. 
If `tiny-cuda-nn` is not installed, the baseline models such as SIREN, NeRF, and DeepONet can still be used.

---

## Requirements

Basic requirements:

```text
numpy
torch
pyyaml
```

Additional requirement for the proposed Instant-NGP-based model:

```text
tinycudann
```

The experiments were conducted using CUDA-enabled GPUs. Exact runtime behavior may depend on the CUDA, PyTorch, and `tiny-cuda-nn` versions.

---

## Data Format

The model input is stored as a NumPy array with shape:

```text
(N, 5)
```

The five columns correspond to:

```text
[r, z, t, J, tau_ns]
```

The model output is stored as a NumPy array with shape:

```text
(N, 3)
```

The three columns correspond to:

```text
[v, p, rho]
```

In the released preprocessed data, the input coordinates are normalized as:

```python
r_norm = r
z_norm = z
t_norm = t / 100
J_norm = log1p(J) / 10
tau_norm = tau_ns / 100
```

The output variables are normalized using a percentile-scaled signed-log min-max normalization scheme. The corresponding preprocessing statistics are provided in:

```text
examples/preprocessing_stats.json
```

---

## Dataset Split

The simulation dataset is split by laser pulse duration and pulse energy.

Two pulse durations are considered:

```text
tau = 5 ns
tau = 10 ns
```

For `tau = 10 ns`, the training energies are:

```text
0.01, 0.03, 0.1, 1.0, 1.5, 2.5 J
```

and the test energies are:

```text
0.05, 0.3, 0.5, 1.3, 2.0, 2.3 J
```

For `tau = 5 ns`, the training energies are:

```text
0.01, 0.05, 0.5, 1.3, 2.0, 2.5 J
```

and the test energies are:

```text
0.03, 0.3, 1.0, 1.5, 2.3 J
```

The test cases are selected at energy levels not included in the training set to evaluate interpolation capability across laser operating regimes.

---

## Models

The repository includes the following models:

| Model | Description |
|---|---|
| `instant_ngp` | Proposed hash-grid-based neural surrogate model |
| `siren` | SIREN baseline |
| `nerf` | NeRF-style coordinate network baseline |
| `deeponet` | DeepONet baseline |

The model architecture is defined in:

```text
src/model.py
```

Model hyperparameters are specified in:

```text
configs/main_experiment.yaml
```

---

## Training

To train the proposed model, run:

```bash
python -m src.train --config configs/main_experiment.yaml
```

The default training configuration uses:

```text
epochs: 10
batch_size: 1024
optimizer: Adam
learning_rate: 1e-4
scheduler: CosineAnnealingLR
loss: MSE
AMP: enabled when CUDA is available
gradient clipping: 1.0
```

Checkpoints are saved to:

```text
checkpoints/
```

---

## Evaluation

The evaluation script computes condition-wise signed-log space RMSE (SLRMSE) using the same aggregation protocol used in the paper.

For each test condition, the evaluation proceeds as follows:

```text
1. Select one tau-J test condition.
2. Compute field-wise RMSE at each time step.
3. Aggregate time-wise RMSE values using RMS.
4. Average the three field-wise scores for v, p, and rho.
```

To evaluate a checkpoint:

```bash
python -m src.evaluate \
  --config configs/main_experiment.yaml \
  --checkpoint checkpoints/PINGP_1_final.pt \
  --output outputs/eval_ingp_seed0.json \
  --csv-output outputs/eval_ingp_seed0.csv
```

The CSV file contains condition-wise errors for each test laser energy and pulse duration.

---

## Evaluating Baseline Models

To evaluate the SIREN baseline:

```bash
python -m src.evaluate \
  --config configs/main_experiment.yaml \
  --model-key siren \
  --checkpoint checkpoints/SIREN_final.pt \
  --output outputs/eval_siren_1_seed0.json \
  --csv-output outputs/eval_siren_seed0.csv
```

To evaluate the NeRF baseline:

```bash
python -m src.evaluate \
  --config configs/main_experiment.yaml \
  --model-key nerf \
  --checkpoint checkpoints/NERF_1_final.pt \
  --output outputs/eval_nerf_seed0.json \
  --csv-output outputs/eval_nerf_seed0.csv
```

To evaluate the DeepONet baseline:

```bash
python -m src.evaluate \
  --config configs/main_experiment.yaml \
  --model-key deeponet \
  --checkpoint checkpoints/DON_1_final.pt \
  --output outputs/eval_deeponet_seed0.json \
  --csv-output outputs/eval_deeponet_seed0.csv
```

To evaluate the proposed model:

```bash
python -m src.evaluate \
  --config configs/main_experiment.yaml \
  --model-key main \
  --checkpoint checkpoints/INGP_1_final.pt \
  --output outputs/eval_ingp_seed0.json \
  --csv-output outputs/eval_ingp_seed0.csv
```

---

## Checkpoints

The `checkpoints/` directory contains trained model weights.

Each model is trained with three random seeds 42, 43, 44

The checkpoint files are expected to contain:

```python
{
    "epoch": ...,
    "model": model.state_dict(),
    "opt": optimizer.state_dict(),
    ...
}
```

---

## Reproducibility

Random seeds are fixed during training. The training script sets seeds for:

```text
Python random
NumPy
PyTorch CPU
PyTorch CUDA
CUDA backend deterministic flags
DataLoader worker initialization
```

However, exact bitwise reproducibility is not always guaranteed due to CUDA kernels, AMP, and `tiny-cuda-nn` operations.

---

## Output Files

Evaluation results are saved in JSON and CSV formats.

Example:

```text
outputs/eval_ingp_seed0.json
outputs/eval_ingp_seed0.csv
```

The CSV output includes:

```text
tau_ns
J
num_time_steps
num_samples
SLRMSE
SLRMSE_v
SLRMSE_p
SLRMSE_rho
global_RMSE_aux
```

The main paper metric is `SLRMSE`.

`global_RMSE_aux` is provided only as an auxiliary metric and is not the primary paper metric.

---

## Notes on Metrics

The main reported metric is SLRMSE in the normalized output space.

For each test condition, field-wise RMSE is first computed at each time step:

```text
RMSE_v(t), RMSE_p(t), RMSE_rho(t)
```

Then the time-dependent errors are aggregated using RMS:

```text
SLRMSE_field = sqrt(mean_t(RMSE_field(t)^2))
```

The final condition-wise score is obtained by averaging over the three physical fields:

```text
SLRMSE = mean(SLRMSE_v, SLRMSE_p, SLRMSE_rho)
```
---

## Inference Demo

An interactive Jupyter notebook version may not be available in this repository.
A rendered HTML demo is provided instead:

- `notebooks/inference_demo.html`

This file shows how to load a trained checkpoint, run inference, and visualize predicted plasma fields.
