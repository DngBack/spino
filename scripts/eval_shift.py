#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ep_operator_dataset import build_case_index, load_case_rollout
from evaluators.predictive_metrics import mae, relative_rmse, rmse
from models.backbones.fno import FNO2d
from models.backbones.pino_fno import PINOFNO2d
from utils.synthetic_solver import simulate_ep2d_from_initial


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint across all shift split families.")
    p.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.v0.2.json"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-type", type=str, default="fno", choices=["fno", "pino"])
    p.add_argument("--output-dir", type=Path, default=Path("outputs/week5_shift_eval"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--short-horizon", type=int, default=20, help="Used for long-rollout split summary.")
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rollout_predict(model: torch.nn.Module, v0: np.ndarray, r0: np.ndarray, horizon: int, device: str) -> np.ndarray:
    cur = np.stack([v0, r0], axis=0)[None, ...]
    outs = [cur[0]]
    model.eval()
    with torch.no_grad():
        for _ in range(horizon - 1):
            x = torch.from_numpy(cur).to(device)
            y = model(x).cpu().numpy()
            cur = y
            outs.append(cur[0])
    return np.stack(outs, axis=0)


def evaluate_case(
    model: torch.nn.Module,
    case: dict[str, Any],
    device: str,
    long_rollout_mode: bool,
    short_horizon: int,
) -> dict[str, float]:
    v = case["V"]
    r = case["R"]
    mask = case["mask"]
    meta = case["metadata"]
    gt = np.stack([v, r], axis=1)
    t_steps = gt.shape[0]

    x = torch.from_numpy(gt[:-1]).to(device)
    y = torch.from_numpy(gt[1:]).to(device)
    with torch.no_grad():
        pred_1 = model(x).cpu().numpy()

    t0 = time.perf_counter()
    pred_roll = rollout_predict(model, v0=v[0], r0=r[0], horizon=t_steps, device=device)
    model_time = time.perf_counter() - t0

    params = meta["parameters"]
    t1 = time.perf_counter()
    _ = simulate_ep2d_from_initial(
        v0=v[0],
        r0=r[0],
        mask=mask,
        diffusion=float(params["diffusion"]) * float(params["conductivity"]),
        excitability=float(params["excitability"]),
        restitution=float(params["restitution"]),
        dt=float(meta["time"]["dt"]),
        num_steps=t_steps,
    )
    solver_time = time.perf_counter() - t1

    out = {
        "one_step_rmse": rmse(pred_1, y.cpu().numpy()),
        "one_step_mae": mae(pred_1, y.cpu().numpy()),
        "one_step_relative_rmse": relative_rmse(pred_1, y.cpu().numpy()),
        "rollout_rmse": rmse(pred_roll, gt),
        "rollout_mae": mae(pred_roll, gt),
        "rollout_relative_rmse": relative_rmse(pred_roll, gt),
        "model_rollout_seconds": model_time,
        "solver_seconds": solver_time,
        "rollout_steps": float(max(t_steps - 1, 1)),
    }

    if long_rollout_mode:
        sh = max(2, min(short_horizon, t_steps))
        out["short_rollout_rmse"] = rmse(pred_roll[:sh], gt[:sh])
        out["long_rollout_rmse"] = rmse(pred_roll, gt)
        out["long_vs_short_ratio"] = out["long_rollout_rmse"] / max(out["short_rollout_rmse"], 1e-8)

    return out


def aggregate(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(metrics[0].keys())
    agg: dict[str, float] = {}
    for k in keys:
        agg[k] = float(np.mean([m[k] for m in metrics]))
    if "rollout_steps" in agg and "model_rollout_seconds" in agg:
        total_steps = sum(m["rollout_steps"] for m in metrics)
        total_mtime = sum(m["model_rollout_seconds"] for m in metrics)
        total_stime = sum(m["solver_seconds"] for m in metrics)
        agg["model_steps_per_second"] = float(total_steps / max(total_mtime, 1e-8))
        agg["solver_steps_per_second"] = float(total_steps / max(total_stime, 1e-8))
    return agg


def select_case_ids(split_payload: dict[str, Any], family: str) -> list[str]:
    if family == "id":
        ids = split_payload.get("test_case_ids", [])
    else:
        ids = split_payload.get("test_shift_case_ids", []) or split_payload.get("test_case_ids", [])
    return list(ids)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields_set = set()
    for r in rows:
        fields_set.update(r.keys())
    fields = sorted(fields_set)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_benchmark_card(manifest: dict[str, Any], split_stats: list[dict[str, Any]]) -> str:
    lines = [
        "# Shift Benchmark Card",
        "",
        f"- Dataset: `{manifest.get('dataset_name')}`",
        f"- Version: `{manifest.get('dataset_version')}`",
        f"- Scenarios: `{manifest.get('scenarios')}`",
        "",
        "## Split Definitions",
        "- `id`: test_case_ids from ID split",
        "- `parameter_shift`: test_shift_case_ids from parameter split (fallback test_case_ids if empty)",
        "- `geometry_shift`: test_shift_case_ids from geometry split (fallback test_case_ids if empty)",
        "- `long_rollout`: evaluates long horizon and reports short-vs-long degradation",
        "",
        "## Case Counts",
    ]
    for st in split_stats:
        lines.append(f"- `{st['family']}`: `{st['num_cases']}` cases")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest)
    split_files = manifest.get("split_files", {})
    case_index = build_case_index(args.manifest)

    model_cls = FNO2d if args.model_type == "fno" else PINOFNO2d
    model = model_cls(
        in_channels=2,
        out_channels=2,
        width=args.width,
        depth=args.depth,
        modes_h=args.modes,
        modes_w=args.modes,
    ).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    family_to_key = {
        "id": "id",
        "parameter_shift": "parameter_shift",
        "geometry_shift": "geometry_shift",
        "long_rollout": "long_rollout",
    }

    aggregate_rows: list[dict[str, Any]] = []
    split_stats: list[dict[str, Any]] = []
    detailed_dir = args.output_dir / "per_split"
    detailed_dir.mkdir(parents=True, exist_ok=True)

    for family, manifest_key in family_to_key.items():
        split_path = Path(split_files[manifest_key])
        split_payload = read_json(split_path)
        case_ids = select_case_ids(split_payload, "id" if family == "id" else family)
        split_stats.append({"family": family, "num_cases": len(case_ids)})
        if not case_ids:
            print(f"[WARN] No cases for split family: {family}")
            continue

        per_case_metrics: list[dict[str, float]] = []
        for cid in tqdm(case_ids, desc=f"Eval {family}"):
            case = load_case_rollout(case_index[cid])
            m = evaluate_case(
                model=model,
                case=case,
                device=args.device,
                long_rollout_mode=(family == "long_rollout"),
                short_horizon=args.short_horizon,
            )
            per_case_metrics.append(m)

        agg = aggregate(per_case_metrics)
        row = {"family": family, "num_cases": len(case_ids), **agg}
        aggregate_rows.append(row)
        write_csv(detailed_dir / f"{family}_per_case_metrics.csv", per_case_metrics)
        (detailed_dir / f"{family}_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    write_csv(args.output_dir / "shift_metrics_table.csv", aggregate_rows)
    (args.output_dir / "shift_metrics_summary.json").write_text(
        json.dumps(aggregate_rows, indent=2), encoding="utf-8"
    )
    (args.output_dir / "shift_benchmark_card.md").write_text(
        make_benchmark_card(manifest, split_stats), encoding="utf-8"
    )

    print(f"[OK] Wrote shift evaluation outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
