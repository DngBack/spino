from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class CaseInfo:
    case_id: str
    tensor_path: Path
    metadata_path: Path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split_case_ids(split_path: Path, split_name: str) -> list[str]:
    split = _load_json(split_path)
    key = f"{split_name}_case_ids"
    if key not in split:
        raise KeyError(f"Split key not found: {key}")
    return split[key]


def build_case_index(manifest_path: Path, repo_root: Path | None = None) -> dict[str, CaseInfo]:
    repo = (repo_root or Path(".")).resolve()
    manifest = _load_json(manifest_path)
    out: dict[str, CaseInfo] = {}
    for case in manifest.get("cases", []):
        case_id = case["case_id"]
        tensor_path = (repo / case["processed_tensor_path"]).resolve()
        metadata_path = (repo / case["metadata_path"]).resolve()
        out[case_id] = CaseInfo(case_id=case_id, tensor_path=tensor_path, metadata_path=metadata_path)
    return out


class EPOneStepDataset(Dataset):
    """
    One-step operator dataset:
    input  x_t = [V_t, R_t], target y_t = [V_{t+1}, R_{t+1}]
    """

    def __init__(
        self,
        manifest_path: Path,
        split_path: Path,
        split_name: str,
        max_cases: int | None = None,
    ) -> None:
        super().__init__()
        self.repo_root = Path(".").resolve()
        self.case_index = build_case_index(manifest_path, self.repo_root)
        case_ids = load_split_case_ids(split_path, split_name)
        if max_cases is not None:
            case_ids = case_ids[:max_cases]
        self.case_ids = case_ids
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._meta_cache: dict[str, dict[str, Any]] = {}
        self.samples: list[tuple[str, int]] = []
        self._build_samples()

    def _load_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        if case_id in self._cache:
            return self._cache[case_id]
        info = self.case_index[case_id]
        data = np.load(info.tensor_path)
        v = data["V"].astype(np.float32)  # [T,H,W]
        r = data["R"].astype(np.float32)  # [T,H,W]
        self._cache[case_id] = (v, r)
        return v, r

    def _build_samples(self) -> None:
        for cid in self.case_ids:
            v, _ = self._load_case(cid)
            t_steps = v.shape[0]
            for t in range(t_steps - 1):
                self.samples.append((cid, t))

    def _load_meta(self, case_id: str) -> dict[str, Any]:
        if case_id in self._meta_cache:
            return self._meta_cache[case_id]
        info = self.case_index[case_id]
        meta = _load_json(info.metadata_path)
        self._meta_cache[case_id] = meta
        return meta

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        case_id, t = self.samples[idx]
        v, r = self._load_case(case_id)
        meta = self._load_meta(case_id)
        info = self.case_index[case_id]
        npz = np.load(info.tensor_path)
        mask = npz["mask"].astype(np.float32)
        params = meta.get("parameters", {})
        diffusion = float(params.get("diffusion", 0.001)) * float(params.get("conductivity", 1.0))
        excitability = float(params.get("excitability", 0.15))
        restitution = float(params.get("restitution", 0.2))
        dt = float(meta.get("time", {}).get("dt", 0.1))
        x = np.stack([v[t], r[t]], axis=0)  # [2,H,W]
        y = np.stack([v[t + 1], r[t + 1]], axis=0)  # [2,H,W]
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(mask),
            "params": torch.tensor([diffusion, excitability, restitution], dtype=torch.float32),
            "dt": torch.tensor(dt, dtype=torch.float32),
            "t_index": torch.tensor(t, dtype=torch.long),
        }


def load_case_rollout(case_info: CaseInfo) -> dict[str, np.ndarray]:
    data = np.load(case_info.tensor_path)
    meta = _load_json(case_info.metadata_path)
    return {
        "V": data["V"].astype(np.float32),
        "R": data["R"].astype(np.float32),
        "mask": data["mask"].astype(np.float32),
        "metadata": meta,  # type: ignore[typeddict-item]
    }
