#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ep_operator_dataset import build_case_index, load_split_case_ids
from losses.physics_loss import pde_residual_magnitude_map
from models.backbones.fno import FNO2d
from models.heads.local_rejector import LocalRejectorCNN
from utils.local_reject_targets import build_patch_targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate local rejector: IoU/Dice + overlay figures.")
    p.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.v0.2.json"))
    p.add_argument("--split", type=Path, default=Path("data/splits/split_v0.2_id.json"))
    p.add_argument("--fno-checkpoint", type=Path, required=True)
    p.add_argument("--local-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/week9_local_eval"))
    p.add_argument("--patch-stride", type=int, default=4, choices=[4, 8])
    p.add_argument("--risk-quantile", type=float, default=0.75)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-viz", type=int, default=6)
    p.add_argument("--split-name", type=str, default="test", choices=["train", "val", "test"])
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    case_index = build_case_index(args.manifest)
    case_ids = load_split_case_ids(args.split, args.split_name)

    fno = FNO2d(in_channels=2, out_channels=2, width=24, depth=3, modes_h=12, modes_w=12).to(device)
    fno.load_state_dict(torch.load(args.fno_checkpoint, map_location=device)["model_state_dict"])
    fno.eval()

    loc = LocalRejectorCNN(in_channels=5, hidden=64, patch_stride=args.patch_stride).to(device)
    loc.load_state_dict(torch.load(args.local_checkpoint, map_location=device)["model_state_dict"])
    loc.eval()

    patch = args.patch_stride
    ious, dices = [], []
    viz_dir = args.output_dir / "overlays"
    viz_dir.mkdir(parents=True, exist_ok=True)
    vcount = 0

    for cid in tqdm(case_ids, desc="Eval local"):
        info = case_index[cid]
        data = np.load(info.tensor_path)
        v = torch.from_numpy(data["V"].astype(np.float32)).to(device)
        r = torch.from_numpy(data["R"].astype(np.float32)).to(device)
        mask_np = data["mask"].astype(np.float32)
        meta = json.loads(info.metadata_path.read_text(encoding="utf-8"))
        params = meta["parameters"]
        diffusion = float(params["diffusion"]) * float(params["conductivity"])
        excitability = float(params["excitability"])
        restitution = float(params["restitution"])
        dt = float(meta["time"]["dt"])
        m = torch.from_numpy(mask_np).to(device)

        T = v.shape[0] - 1
        times = [0, T // 2, T - 1] if T > 2 else [0]

        for t in times:
            x = torch.stack([v[t], r[t]], dim=0).unsqueeze(0)
            y = torch.stack([v[t + 1], r[t + 1]], dim=0).unsqueeze(0)
            params_t = torch.tensor([[diffusion, excitability, restitution]], device=device)
            dt_t = torch.tensor([dt], device=device)

            pred = fno(x)
            res_mag = pde_residual_magnitude_map(x, pred, params_t, dt_t, m.unsqueeze(0))
            inp = torch.cat([x, pred, res_mag], dim=1)
            logits = loc(inp)
            probs = torch.sigmoid(logits)
            pred_m = (probs > 0.5).float()
            tgt = build_patch_targets(pred, y, m.unsqueeze(0), patch, args.risk_quantile)
            if logits.shape[-2:] != tgt.shape[-2:]:
                tgt = F.interpolate(tgt, size=logits.shape[-2:], mode="nearest")

            inter = (pred_m * tgt).sum()
            union = ((pred_m + tgt) > 0).float().sum()
            iou = float((inter / (union + 1e-8)).item())
            dice = float((2 * inter / (pred_m.sum() + tgt.sum() + 1e-8)).item())
            ious.append(iou)
            dices.append(dice)

            if vcount < args.num_viz:
                H, W = v.shape[-2], v.shape[-1]
                prob_up = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)[
                    0, 0
                ].cpu().numpy()
                vt = v[t].cpu().numpy()
                gt_err = (pred - y).abs().mean(dim=1)[0].cpu().numpy()

                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                axes[0].imshow(vt * mask_np, cmap="coolwarm")
                axes[0].set_title("V(t)")
                axes[1].imshow(gt_err * mask_np, cmap="magma")
                axes[1].set_title("|pred-gt|")
                axes[2].imshow(prob_up * mask_np, cmap="viridis", vmin=0, vmax=1)
                axes[2].set_title("risk prob")
                axes[3].imshow((prob_up > 0.5).astype(float) * mask_np, cmap="Greys")
                axes[3].set_title("defer (patch)")
                for ax in axes:
                    ax.axis("off")
                fig.suptitle(f"{cid} t={t}")
                fig.tight_layout()
                fig.savefig(viz_dir / f"viz_{vcount:03d}.png", dpi=160)
                plt.close(fig)
                vcount += 1

    oracle_note = (
        "Oracle patch mask equals supervision target (GT-derived); perfect predictor would achieve IoU=1 vs that target."
    )
    metrics = {
        "split": args.split_name,
        "num_eval_points": len(ious),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "patch_stride": patch,
        "risk_quantile": args.risk_quantile,
        "oracle_note": oracle_note,
    }
    (args.output_dir / "local_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "week9_exit_check.md").write_text(
        "\n".join(
            [
                "# Week 9 Local Rejector Exit Check",
                "",
                f"- Mean IoU (pred mask vs GT-derived unsafe patches): `{metrics['mean_iou']:.4f}`",
                f"- Mean Dice: `{metrics['mean_dice']:.4f}`",
                "",
                oracle_note,
                "",
                f"- Overlays: `{viz_dir}`",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"[OK] Overlays: {viz_dir}")


if __name__ == "__main__":
    main()
