#!/usr/bin/env python3
"""
Generate a synthetic 2D EP-like dataset for SPINO Week 2 deliverables.

The generated trajectories are lightweight reaction-diffusion style simulations
with scenario-specific stimulus and dynamics settings. This is intended as a
practical dataset bootstrap for baseline operator training and evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCENARIOS = ["planar_wave", "centrifugal_wave", "stable_spiral", "spiral_breakup"]
SCENARIO_SEED_BIAS = {name: i * 5_000_000 for i, name in enumerate(SCENARIOS)}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class SimConfig:
    dt: float
    dx: float
    num_steps: int
    h: int
    w: int
    diffusion: float
    excitability: float
    restitution: float
    stimulus_amplitude: float
    stimulus_duration: int
    start_step: int
    scenario: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic EP2D dataset + case metadata.")
    parser.add_argument("--dataset-version", type=str, default="v0.2")
    parser.add_argument("--output-root", type=Path, default=Path("."), help="Repo root (data/ is created under here).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Alias for --output-root (same as passing repo root where data/ lives).",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Paper-oriented defaults: ~96 cases per scenario (~384 total), 2 resolutions, longer horizon.",
    )
    parser.add_argument("--num-parameter-sets", type=int, default=3)
    parser.add_argument("--num-seeds-per-parameter-set", type=int, default=4)
    parser.add_argument("--num-geometries", type=int, default=3)
    parser.add_argument(
        "--cases-per-scenario",
        type=int,
        default=None,
        help="Cap exactly N cases per scenario (deterministic grid). Automatically expands param sets if needed.",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=None,
        help="Total cases across all scenarios; sets cases-per-scenario = ceil(num_cases / len(SCENARIOS)).",
    )
    parser.add_argument("--num-steps", type=int, default=120)
    parser.add_argument("--resolutions", type=str, default="64,96")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--global-seed", type=int, default=20260409)
    parser.add_argument("--clean", action="store_true", help="Remove previous generated synthetic dataset artifacts.")
    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_root = args.output_dir
    if args.paper:
        args.dataset_version = "v1.0"
        args.num_steps = max(args.num_steps, 160)
        args.num_geometries = max(args.num_geometries, 8)
        args.num_seeds_per_parameter_set = max(args.num_seeds_per_parameter_set, 5)
        if args.cases_per_scenario is None and args.num_cases is None:
            args.cases_per_scenario = 96
    if args.num_cases is not None:
        n = max(1, int(args.num_cases))
        args.cases_per_scenario = max(1, (n + len(SCENARIOS) - 1) // len(SCENARIOS))
    return args


def build_case_slot_list(
    num_parameter_sets: int,
    num_geometries: int,
    num_seeds: int,
    resolutions: list[int],
    cases_per_scenario: int | None,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """
    Deterministic (pidx, gidx, ridx, sidx) tuples. If cases_per_scenario is set, expands
    num_parameter_sets until the full grid has at least that many slots, then truncates.
    """
    n_res = len(resolutions)

    def grid_for_P(P: int) -> list[tuple[int, int, int, int]]:
        g: list[tuple[int, int, int, int]] = []
        for pidx in range(1, P + 1):
            for gidx in range(1, num_geometries + 1):
                for ridx in range(n_res):
                    for sidx in range(num_seeds):
                        g.append((pidx, gidx, ridx, sidx))
        return g

    P = max(1, num_parameter_sets)
    slots = grid_for_P(P)
    if cases_per_scenario is not None:
        need = int(cases_per_scenario)
        while len(slots) < need:
            P += 1
            slots = grid_for_P(P)
        slots = slots[:need]
    return slots, P


def laplacian(z: np.ndarray) -> np.ndarray:
    return (
        np.roll(z, 1, axis=0)
        + np.roll(z, -1, axis=0)
        + np.roll(z, 1, axis=1)
        + np.roll(z, -1, axis=1)
        - 4.0 * z
    )


def make_geometry_mask(h: int, w: int, geometry_id: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cy = h * (0.5 + 0.08 * np.sin(geometry_id))
    cx = w * (0.5 + 0.08 * np.cos(geometry_id))
    ry = h * (0.43 + 0.03 * (geometry_id % 3))
    rx = w * (0.43 + 0.02 * ((geometry_id + 1) % 3))
    ellipse = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0

    # Create a deterministic notch to vary geometry.
    notch_w = max(2, int(w * 0.05))
    notch_h = max(2, int(h * (0.12 + 0.03 * (geometry_id % 2))))
    notch_x0 = int(w * (0.15 + 0.1 * ((geometry_id + 2) % 3)))
    notch_y0 = int(h * 0.0)
    ellipse[notch_y0 : notch_y0 + notch_h, notch_x0 : notch_x0 + notch_w] = False

    # Small random erosion-like dropout for variability.
    dropout = rng.uniform(size=(h, w)) < (0.002 * (1 + geometry_id % 2))
    mask = np.logical_and(ellipse, ~dropout)
    return mask.astype(np.float32)


def apply_stimulus(v: np.ndarray, step: int, cfg: SimConfig, rng: np.random.Generator) -> None:
    h, w = v.shape
    if not (cfg.start_step <= step < cfg.start_step + cfg.stimulus_duration):
        return

    if cfg.scenario == "planar_wave":
        v[: max(2, h // 12), :] += cfg.stimulus_amplitude
    elif cfg.scenario == "centrifugal_wave":
        cy, cx = h // 2, w // 2
        rr = max(2, min(h, w) // 12)
        yy, xx = np.mgrid[0:h, 0:w]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= rr**2
        v[disk] += cfg.stimulus_amplitude
    elif cfg.scenario == "stable_spiral":
        v[h // 2 - 2 : h // 2 + 2, : max(2, w // 10)] += cfg.stimulus_amplitude
        if step == cfg.start_step + 2:
            v[: max(2, h // 8), w // 2 - 2 : w // 2 + 2] += 0.8 * cfg.stimulus_amplitude
    elif cfg.scenario == "spiral_breakup":
        y = rng.integers(low=max(2, h // 8), high=max(3, h - h // 8))
        x = rng.integers(low=max(2, w // 8), high=max(3, w - w // 8))
        rr = max(2, min(h, w) // 14)
        yy, xx = np.mgrid[0:h, 0:w]
        disk = (yy - y) ** 2 + (xx - x) ** 2 <= rr**2
        v[disk] += cfg.stimulus_amplitude * rng.uniform(0.8, 1.2)


def simulate_case(cfg: SimConfig, geometry_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    h, w = cfg.h, cfg.w
    v = np.zeros((h, w), dtype=np.float32)
    r = np.zeros((h, w), dtype=np.float32)
    traj_v = np.zeros((cfg.num_steps, h, w), dtype=np.float32)
    traj_r = np.zeros((cfg.num_steps, h, w), dtype=np.float32)

    # Scenario-specific turbulence/noise for harder dynamics.
    noise_scale = 0.003 if cfg.scenario in ("planar_wave", "centrifugal_wave") else 0.008
    breakup_gain = 1.25 if cfg.scenario == "spiral_breakup" else 1.0

    for t in range(cfg.num_steps):
        apply_stimulus(v, t, cfg, rng)
        lap = laplacian(v)
        reaction = cfg.excitability * v * (1.0 - v) * (v - 0.08) - r
        recov = cfg.restitution * (v - 0.15 * r)

        v_next = v + cfg.dt * (cfg.diffusion * lap + reaction)
        r_next = r + cfg.dt * recov
        v_next += noise_scale * breakup_gain * rng.normal(size=(h, w)).astype(np.float32)

        v_next *= geometry_mask
        r_next *= geometry_mask
        v_next = np.clip(v_next, -1.0, 1.5)
        r_next = np.clip(r_next, -1.0, 2.0)

        v, r = v_next.astype(np.float32), r_next.astype(np.float32)
        traj_v[t] = v
        traj_r[t] = r

    return traj_v, traj_r


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def maybe_clean(base: Path) -> None:
    # Safe cleanup for known generated folders only.
    targets = [
        base / "data/raw/synthetic_runs",
        base / "data/processed/tensors",
        base / "data/metadata/case_metadata",
    ]
    for p in targets:
        if not p.exists():
            continue
        for fp in p.rglob("*"):
            if fp.is_file():
                fp.unlink()


def main() -> None:
    args = parse_args()
    base = args.output_root.resolve()
    rng = np.random.default_rng(args.global_seed)
    if args.clean:
        maybe_clean(base)

    resolutions = [int(x.strip()) for x in args.resolutions.split(",") if x.strip()]
    raw_root = base / "data/raw/synthetic_runs"
    processed_root = base / "data/processed/tensors"
    case_meta_root = base / "data/metadata/case_metadata"
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    case_meta_root.mkdir(parents=True, exist_ok=True)

    all_case_ids: list[str] = []
    slots, effective_P = build_case_slot_list(
        args.num_parameter_sets,
        args.num_geometries,
        args.num_seeds_per_parameter_set,
        resolutions,
        args.cases_per_scenario,
    )

    generation_meta = {
        "dataset_name": "spino_ep2d",
        "dataset_version": args.dataset_version,
        "generator": "synthetic",
        "global_seed": args.global_seed,
        "num_parameter_sets_requested": args.num_parameter_sets,
        "effective_num_parameter_sets": effective_P,
        "num_seeds_per_parameter_set": args.num_seeds_per_parameter_set,
        "num_geometries": args.num_geometries,
        "cases_per_scenario": args.cases_per_scenario,
        "num_steps": args.num_steps,
        "resolutions": resolutions,
        "scenarios": SCENARIOS,
        "slots_per_scenario": len(slots),
        "expected_total_cases": len(SCENARIOS) * len(slots),
        "created_at": now_iso(),
    }

    for scenario in SCENARIOS:
        params_by_p: dict[int, tuple[float, float, float, float]] = {}
        for pidx, _, _, _ in slots:
            if pidx not in params_by_p:
                params_by_p[pidx] = (
                    float(rng.uniform(0.0008, 0.0014)),
                    float(rng.uniform(0.8, 1.2)),
                    float(rng.uniform(0.10, 0.24)),
                    float(rng.uniform(0.10, 0.35)),
                )

        for pidx, gidx, ridx, sidx in slots:
            diffusion, conductivity, excitability, restitution = params_by_p[pidx]
            res = resolutions[ridx]
            seed = int(
                args.global_seed
                + SCENARIO_SEED_BIAS[scenario]
                + 10_000 * pidx
                + 1_000 * gidx
                + 100 * ridx
                + sidx
            )
            cfg = SimConfig(
                dt=args.dt,
                dx=1.0 / float(res),
                num_steps=args.num_steps,
                h=res,
                w=res,
                diffusion=diffusion * conductivity,
                excitability=excitability,
                restitution=restitution,
                stimulus_amplitude=float(rng.uniform(0.8, 1.2)),
                stimulus_duration=10,
                start_step=5,
                scenario=scenario,
                seed=seed,
            )

            geom_rng = np.random.default_rng(seed + 77)
            geom_mask = make_geometry_mask(res, res, gidx, geom_rng)
            traj_v, traj_r = simulate_case(cfg, geom_mask)

            case_id = (
                f"spino_{scenario}_geo{gidx:02d}_p{pidx:02d}_s{seed}" f"_r{res}_t{args.num_steps}"
            )
            tensor_path = processed_root / f"{case_id}.npz"
            np.savez_compressed(
                tensor_path,
                V=traj_v,
                R=traj_r,
                mask=geom_mask,
                V0=traj_v[0],
                R0=traj_r[0],
            )
            tensor_hash = sha256_file(tensor_path)

            raw_case_path = raw_root / scenario / f"geo{gidx:02d}" / f"p{pidx:02d}" / f"s{seed}"
            raw_case_path.mkdir(parents=True, exist_ok=True)

            case_meta = {
                "case_id": case_id,
                "dataset_version": args.dataset_version,
                "generator": "synthetic",
                "scenario_type": scenario,
                "geometry_id": f"geo{gidx:02d}",
                "parameter_set_id": f"p{pidx:02d}",
                "mesh_or_grid": {
                    "type": "grid",
                    "shape": [res, res],
                    "resolution_tag": f"r{res}",
                    "dx": cfg.dx,
                },
                "time": {"dt": cfg.dt, "num_steps": cfg.num_steps, "t_start": 0.0},
                "seed": seed,
                "parameters": {
                    "diffusion": diffusion,
                    "conductivity": conductivity,
                    "excitability": excitability,
                    "restitution": restitution,
                },
                "stimulus": {
                    "type": "scenario_defined",
                    "location": None,
                    "start_step": cfg.start_step,
                    "duration_steps": cfg.stimulus_duration,
                    "amplitude": cfg.stimulus_amplitude,
                },
                "state_channels": {"input_channels": ["V0", "R0"], "output_channels": ["V", "R"]},
                "file_paths": {
                    "raw_case_path": str(raw_case_path.relative_to(base)),
                    "processed_tensor_path": str(tensor_path.relative_to(base)),
                },
                "split_tags": ["unassigned"],
                "quality_flags": {
                    "has_nan": bool(np.isnan(traj_v).any() or np.isnan(traj_r).any()),
                    "has_inf": bool(np.isinf(traj_v).any() or np.isinf(traj_r).any()),
                    "is_duplicate_hash": False,
                    "physics_warning": False,
                },
                "hashes": {"processed_tensor_sha256": tensor_hash},
                "created_at": now_iso(),
            }
            save_json(case_meta_root / f"{case_id}.json", case_meta)
            all_case_ids.append(case_id)

    gen_cfg_path = base / f"data/metadata/generation_config.{args.dataset_version}.json"
    save_json(gen_cfg_path, generation_meta)
    print(f"[OK] Generated cases: {len(all_case_ids)}")
    print(f"[OK] Case metadata dir: {case_meta_root}")
    print(f"[OK] Tensors dir: {processed_root}")
    print(f"[OK] Generation config: {gen_cfg_path}")


if __name__ == "__main__":
    main()
