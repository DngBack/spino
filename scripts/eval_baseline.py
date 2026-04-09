#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ep_operator_dataset import build_case_index, load_case_rollout, load_split_case_ids
from evaluators.predictive_metrics import mae, relative_rmse, rmse
from models.backbones.fno import FNO2d
from utils.synthetic_solver import simulate_ep2d_from_initial


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Week-3 FNO baseline.")
    p.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.v0.2.json"))
    p.add_argument("--split", type=Path, default=Path("data/splits/split_v0.2_id.json"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/week3_fno_eval"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--num-qual-cases", type=int, default=3)
    return p.parse_args()


def rollout_predict(model: torch.nn.Module, v0: np.ndarray, r0: np.ndarray, horizon: int, device: str) -> np.ndarray:
    model.eval()
    cur = np.stack([v0, r0], axis=0)[None, ...]  # [1,2,H,W]
    preds = [cur[0]]
    with torch.no_grad():
        for _ in range(horizon - 1):
            x = torch.from_numpy(cur).to(device)
            y = model(x).cpu().numpy()
            cur = y
            preds.append(cur[0])
    return np.stack(preds, axis=0)  # [T,2,H,W]


def save_qualitative_plot(case_id: str, gt: np.ndarray, pred: np.ndarray, out_path: Path) -> None:
    # show V channel at 3 times
    times = [0, gt.shape[0] // 2, gt.shape[0] - 1]
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, t in enumerate(times):
        g = gt[t, 0]
        p = pred[t, 0]
        d = np.abs(g - p)
        axes[i, 0].imshow(g, cmap="coolwarm")
        axes[i, 0].set_title(f"GT V t={t}")
        axes[i, 1].imshow(p, cmap="coolwarm")
        axes[i, 1].set_title(f"Pred V t={t}")
        axes[i, 2].imshow(d, cmap="magma")
        axes[i, 2].set_title(f"|Err| t={t}")
        for j in range(3):
            axes[i, j].axis("off")
    fig.suptitle(case_id)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_index = build_case_index(args.manifest)
    test_ids = load_split_case_ids(args.split, "test")

    model = FNO2d(
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

    one_step_rmse = []
    one_step_mae = []
    one_step_rrmse = []
    rollout_rmse = []
    rollout_mae = []
    rollout_rrmse = []
    per_horizon_rmse: list[list[float]] = []

    model_rollout_seconds = 0.0
    solver_seconds = 0.0
    rollout_steps = 0

    for idx, cid in enumerate(tqdm(test_ids, desc="Eval test cases")):
        case = load_case_rollout(case_index[cid])
        v = case["V"]  # [T,H,W]
        r = case["R"]
        mask = case["mask"]
        meta = case["metadata"]
        gt = np.stack([v, r], axis=1)  # [T,2,H,W]

        # One-step from all frames
        x = torch.from_numpy(gt[:-1]).to(args.device)
        y = torch.from_numpy(gt[1:]).to(args.device)
        with torch.no_grad():
            pred_1 = model(x).cpu().numpy()
        y_np = y.cpu().numpy()
        one_step_rmse.append(rmse(pred_1, y_np))
        one_step_mae.append(mae(pred_1, y_np))
        one_step_rrmse.append(relative_rmse(pred_1, y_np))

        # Autoregressive rollout
        t0 = time.perf_counter()
        pred_roll = rollout_predict(model, v0=v[0], r0=r[0], horizon=v.shape[0], device=args.device)
        model_rollout_seconds += time.perf_counter() - t0
        rollout_steps += int(v.shape[0] - 1)

        rr = rmse(pred_roll, gt)
        ra = mae(pred_roll, gt)
        rrr = relative_rmse(pred_roll, gt)
        rollout_rmse.append(rr)
        rollout_mae.append(ra)
        rollout_rrmse.append(rrr)
        per_horizon_rmse.append(
            [rmse(pred_roll[t], gt[t]) for t in range(gt.shape[0])]
        )

        # Reference solver runtime
        params = meta["parameters"]
        dt = float(meta["time"]["dt"])
        t1 = time.perf_counter()
        _ = simulate_ep2d_from_initial(
            v0=v[0],
            r0=r[0],
            mask=mask,
            diffusion=float(params["diffusion"]) * float(params["conductivity"]),
            excitability=float(params["excitability"]),
            restitution=float(params["restitution"]),
            dt=dt,
            num_steps=int(v.shape[0]),
        )
        solver_seconds += time.perf_counter() - t1

        if idx < args.num_qual_cases:
            save_qualitative_plot(
                case_id=cid,
                gt=gt,
                pred=pred_roll,
                out_path=args.output_dir / "qualitative" / f"{cid}.png",
            )

    horizon_rmse = np.mean(np.array(per_horizon_rmse), axis=0)
    fig = plt.figure(figsize=(7, 4))
    plt.plot(horizon_rmse)
    plt.title("Rollout RMSE Across Horizon (ID test)")
    plt.xlabel("Time step")
    plt.ylabel("RMSE")
    plt.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "rollout_rmse_horizon.png", dpi=180)
    plt.close(fig)

    metrics = {
        "split": "id_test",
        "num_test_cases": len(test_ids),
        "one_step_rmse": float(np.mean(one_step_rmse)),
        "one_step_mae": float(np.mean(one_step_mae)),
        "one_step_relative_rmse": float(np.mean(one_step_rrmse)),
        "rollout_rmse": float(np.mean(rollout_rmse)),
        "rollout_mae": float(np.mean(rollout_mae)),
        "rollout_relative_rmse": float(np.mean(rollout_rrmse)),
        "model_rollout_seconds_total": float(model_rollout_seconds),
        "solver_seconds_total": float(solver_seconds),
        "model_steps_per_second": float(rollout_steps / max(model_rollout_seconds, 1e-8)),
        "solver_steps_per_second": float(rollout_steps / max(solver_seconds, 1e-8)),
    }

    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    write_metrics_csv(args.output_dir / "baseline_metrics_id.csv", metrics)
    print(json.dumps(metrics, indent=2))
    print(f"[OK] Saved evaluation artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
