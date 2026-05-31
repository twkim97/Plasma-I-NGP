
from __future__ import annotations

import math
from typing import Any, Dict, Literal

import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None


class FourierPE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_frequencies: int = 10,
        include_input: bool = True,
        pe_type: Literal["fourier", "none"] = "fourier",
    ):
        super().__init__()

        self.in_dim = in_dim
        self.n_frequencies = n_frequencies
        self.include_input = include_input
        self.pe_type = pe_type
        self.fourier_scale = 2.0 * math.pi

        if pe_type == "fourier":
            self.out_dim = (in_dim if include_input else 0) + 2 * n_frequencies * in_dim
        elif pe_type == "none":
            self.out_dim = in_dim
        else:
            raise ValueError(f"Unsupported pe_type: {pe_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pe_type == "none":
            return x

        freqs = 2.0 ** torch.arange(
            self.n_frequencies,
            device=x.device,
            dtype=x.dtype,
        )

        xb = x.unsqueeze(-1) * freqs.view(1, 1, -1) * self.fourier_scale
        sin = torch.sin(xb).reshape(x.shape[0], -1)
        cos = torch.cos(xb).reshape(x.shape[0], -1)

        if self.include_input:
            return torch.cat([x, sin, cos], dim=-1)

        return torch.cat([sin, cos], dim=-1)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        activation: Literal["relu", "silu", "gelu"] = "relu",
    ):
        super().__init__()

        act_layer = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }[activation]

        layers = []
        last_dim = in_dim

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act_layer())
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlasmaInstantNGP(nn.Module):
    def __init__(
        self,
        out_dim: int = 3,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
        n_levels: int = 12,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 16,
        base_resolution: int = 16,
        per_level_scale: float = 1.3,
        interpolation: str = "Smoothstep",
    ):
        super().__init__()

        if tcnn is None:
            raise ImportError(
                "PlasmaInstantNGP requires tiny-cuda-nn. "
                "Please install tinycudann or use a baseline model."
            )

        self.hash_enc = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
                "interpolation": interpolation,
            },
        )

        self.t_reparam = tcnn.Network(
            n_input_dims=3,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim,
                "n_hidden_layers": n_hidden_layers,
            },
        )

        self.trunk = tcnn.Network(
            n_input_dims=self.hash_enc.n_output_dims + 3,
            n_output_dims=out_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": n_hidden_layers,
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 5) = [r, z, t, J, tau_ns]
        y: (N, 3) = [v, p, rho]
        """
        rz = x[:, 0:2]
        tJtau = x[:, 2:5]

        t_reparam = self.t_reparam(tJtau)
        rzt_reparam = torch.cat([rz, t_reparam], dim=-1)

        hash_feat = self.hash_enc(rzt_reparam)
        trunk_input = torch.cat([hash_feat, tJtau], dim=-1)

        return self.trunk(trunk_input)


class SirenLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_dim, out_dim)
        self._init_weights(in_dim)

    def _init_weights(self, in_dim: int):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_dim
            else:
                bound = math.sqrt(6.0 / in_dim) / self.omega_0

            self.linear.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class PlasmaSiren(nn.Module):
    def __init__(
        self,
        in_dim: int = 5,
        out_dim: int = 3,
        hidden_dim: int = 1024,
        n_hidden_layers: int = 8,
        outer_omega_0: float = 30.0,
        inner_omega_0: float = 30.0,
    ):
        super().__init__()

        layers = [
            SirenLayer(
                in_dim=in_dim,
                out_dim=hidden_dim,
                is_first=True,
                omega_0=outer_omega_0,
            )
        ]

        for _ in range(n_hidden_layers - 1):
            layers.append(
                SirenLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    is_first=False,
                    omega_0=inner_omega_0,
                )
            )

        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(hidden_dim, out_dim)
        self._init_final_weights(hidden_dim, inner_omega_0)

    def _init_final_weights(self, hidden_dim: int, inner_omega_0: float):
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / inner_omega_0
            self.final_linear.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.final_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_linear(self.net(x))


class PlasmaNeRF(nn.Module):
    def __init__(
        self,
        out_dim: int = 3,
        n_freq_pos: int = 10,
        n_freq_cond: int = 4,
        trunk_width: int = 512,
        trunk_depth: int = 5,
        skip_at: int = 4,
        head_width: int = 128,
        head_depth: int = 1,
    ):
        super().__init__()

        self.pos_pe = FourierPE(
            in_dim=3,
            n_frequencies=n_freq_pos,
            include_input=True,
            pe_type="fourier",
        )

        self.cond_pe = FourierPE(
            in_dim=2,
            n_frequencies=n_freq_cond,
            include_input=True,
            pe_type="fourier",
        )

        self.skip_at = skip_at
        self.trunk = nn.ModuleList()

        self.trunk.append(nn.Linear(self.pos_pe.out_dim, trunk_width))

        for i in range(1, trunk_depth):
            layer_in_dim = trunk_width + self.pos_pe.out_dim if i == skip_at else trunk_width
            self.trunk.append(nn.Linear(layer_in_dim, trunk_width))

        self.head = MLP(
            in_dim=trunk_width + self.cond_pe.out_dim,
            out_dim=out_dim,
            hidden_dim=head_width,
            n_hidden_layers=head_depth,
            activation="relu",
        )

        self.act = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rzt = x[:, 0:3]
        Jtau = x[:, 3:5]

        pos = self.pos_pe(rzt)
        cond = self.cond_pe(Jtau)

        h = pos

        for i, layer in enumerate(self.trunk):
            if i == self.skip_at:
                h = torch.cat([h, pos], dim=-1)

            h = self.act(layer(h))

        return self.head(torch.cat([h, cond], dim=-1))


class PlasmaDeepONet(nn.Module):
    def __init__(
        self,
        out_dim: int = 3,
        hidden_dim: int = 512,
        rank: int = 1024,
        n_hidden_layers: int = 3,
        trunk_n_frequencies: int = 10,
        branch_pe_type: Literal["fourier", "none"] = "none",
        branch_n_frequencies: int = 4,
    ):
        super().__init__()

        self.trunk_pe = FourierPE(
            in_dim=3,
            n_frequencies=trunk_n_frequencies,
            include_input=True,
            pe_type="fourier",
        )

        self.branch_pe = FourierPE(
            in_dim=2,
            n_frequencies=branch_n_frequencies,
            include_input=True,
            pe_type=branch_pe_type,
        )

        self.trunk = MLP(
            in_dim=self.trunk_pe.out_dim,
            out_dim=rank,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            activation="relu",
        )

        self.branch = MLP(
            in_dim=self.branch_pe.out_dim,
            out_dim=rank,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            activation="relu",
        )

        self.head = MLP(
            in_dim=rank,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=1,
            activation="relu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rzt = x[:, 0:3]
        Jtau = x[:, 3:5]

        trunk_feat = self.trunk(self.trunk_pe(rzt))
        branch_feat = self.branch(self.branch_pe(Jtau))

        return self.head(trunk_feat * branch_feat)


MODEL_REGISTRY = {
    "instant_ngp": PlasmaInstantNGP,
    "siren": PlasmaSiren,
    "nerf": PlasmaNeRF,
    "deeponet": PlasmaDeepONet,
}


def build_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    Build a model from a configuration dictionary.

    Example:
        model_config = {
            "name": "instant_ngp",
            "params": {
                "hidden_dim": 64,
                "n_hidden_layers": 2,
                "n_levels": 16,
                "log2_hashmap_size": 19
            }
        }

        model = build_model(model_config)
    """
    if "name" not in model_config:
        raise KeyError("model_config must contain the key 'name'.")

    model_name = model_config["name"]
    model_params = model_config.get("params", {})

    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model name: {model_name}. Available models: {available}")

    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(**model_params)