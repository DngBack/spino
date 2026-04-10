#!/usr/bin/env python3
"""
Build ID + shift split files from dataset manifest.

Example:
  python scripts/build_splits.py \
    --manifest data/metadata/dataset_manifest.v1.0.json \
    --output-dir data/splits \
    --dataset-version v1.0 \
    --seed 42 \
    --param-key excitability \
    --param-threshold 0.18 \
    --geometry-holdout-fraction 0.2
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset split files from manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/dataset_manifest.v1.0.json"),
        help="Path to dataset manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory for split JSON outputs.",
    )
    parser.add_argument("--dataset-version", type=str, default="v1.0", help="Dataset version in split files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split creation.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="ID split train ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="ID split validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="ID split test ratio.")
    parser.add_argument(
        "--param-key",
        type=str,
        default="excitability",
        help="Parameter key used for parameter-shift split.",
    )
    parser.add_argument(
        "--param-threshold",
        type=float,
        default=0.18,
        help="Threshold for parameter-shift split: <= threshold in train, > threshold in shift test.",
    )
    parser.add_argument(
        "--geometry-holdout-fraction",
        type=float,
        default=0.20,
        help="Fraction of geometry IDs held out for geometry-shift test.",
    )
    return parser.parse_args()


def ratio_check(args: argparse.Namespace) -> None:
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"train/val/test ratios must sum to 1.0; got {total}")


def stratified_id_split(cases: list[dict[str, Any]], rng: random.Random, train_r: float, val_r: float) -> tuple[list[str], list[str], list[str]]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        by_scenario[c.get("scenario_type", "unknown")].append(c)

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for _, bucket in by_scenario.items():
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(round(n * train_r))
        n_val = int(round(n * val_r))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val

        train_ids.extend(c["case_id"] for c in bucket[:n_train])
        val_ids.extend(c["case_id"] for c in bucket[n_train : n_train + n_val])
        test_ids.extend(c["case_id"] for c in bucket[n_train + n_val : n_train + n_val + n_test])

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def make_split_payload(
    split_name: str,
    dataset_version: str,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    notes: list[str],
    test_shift_ids: list[str] | None = None,
    shift_definition: str = "",
) -> dict[str, Any]:
    return {
        "split_name": split_name,
        "dataset_version": dataset_version,
        "created_at": now_iso(),
        "train_case_ids": sorted(train_ids),
        "val_case_ids": sorted(val_ids),
        "test_case_ids": sorted(test_ids),
        "test_shift_case_ids": sorted(test_shift_ids or []),
        "shift_definition": shift_definition,
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    ratio_check(args)

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    manifest = load_json(args.manifest)
    cases = manifest.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest contains no cases.")

    rng = random.Random(args.seed)
    case_by_id = {c["case_id"]: c for c in cases if "case_id" in c}
    all_case_ids = sorted(case_by_id.keys())

    # 1) ID split (stratified by scenario_type)
    id_train, id_val, id_test = stratified_id_split(cases, rng, args.train_ratio, args.val_ratio)
    split_id = make_split_payload(
        split_name=f"split_{args.dataset_version}_id",
        dataset_version=args.dataset_version,
        train_ids=id_train,
        val_ids=id_val,
        test_ids=id_test,
        notes=["Stratified by scenario_type", f"seed={args.seed}"],
    )

    # 2) Parameter shift split
    # Requires metadata with parameter key in case metadata; we infer from metadata file.
    shift_train: list[str] = []
    shift_test: list[str] = []
    shift_val: list[str] = []
    repo_root = Path(".").resolve()
    for case_id in all_case_ids:
        case = case_by_id[case_id]
        meta_path = repo_root / case.get("metadata_path", "")
        if not meta_path.exists():
            # keep conservative: route missing metadata to non-shift test bucket
            shift_test.append(case_id)
            continue
        meta = load_json(meta_path)
        val = meta.get("parameters", {}).get(args.param_key)
        if val is None:
            shift_test.append(case_id)
            continue
        if float(val) > args.param_threshold:
            shift_test.append(case_id)
        else:
            shift_train.append(case_id)
    rng.shuffle(shift_train)
    n_val = int(round(len(shift_train) * args.val_ratio))
    shift_val = sorted(shift_train[:n_val])
    shift_train = sorted(shift_train[n_val:])
    split_param = make_split_payload(
        split_name=f"split_{args.dataset_version}_param_shift",
        dataset_version=args.dataset_version,
        train_ids=shift_train,
        val_ids=shift_val,
        test_ids=[],
        test_shift_ids=shift_test,
        shift_definition=f"{args.param_key} <= {args.param_threshold} for train/val, > threshold for test_shift",
        notes=["Parameter shift split", f"seed={args.seed}"],
    )

    # 3) Geometry shift split
    geometry_to_ids: dict[str, list[str]] = defaultdict(list)
    for c in cases:
        geometry_to_ids[c.get("geometry_id", "unknown")].append(c["case_id"])
    geometry_ids = sorted(geometry_to_ids.keys())
    rng.shuffle(geometry_ids)
    num_holdout = max(1, int(round(len(geometry_ids) * args.geometry_holdout_fraction)))
    holdout_geometries = set(geometry_ids[:num_holdout])
    geo_shift_ids = sorted(
        case_id for g, ids in geometry_to_ids.items() if g in holdout_geometries for case_id in ids
    )
    geo_pool = [case_id for case_id in all_case_ids if case_id not in set(geo_shift_ids)]
    rng.shuffle(geo_pool)
    geo_n_val = int(round(len(geo_pool) * args.val_ratio))
    geo_train = sorted(geo_pool[geo_n_val:])
    geo_val = sorted(geo_pool[:geo_n_val])
    split_geo = make_split_payload(
        split_name=f"split_{args.dataset_version}_geometry_shift",
        dataset_version=args.dataset_version,
        train_ids=geo_train,
        val_ids=geo_val,
        test_ids=[],
        test_shift_ids=geo_shift_ids,
        shift_definition=f"Holdout geometry_ids fraction={args.geometry_holdout_fraction}",
        notes=["Geometry shift split", f"holdout_geometry_ids={sorted(holdout_geometries)}", f"seed={args.seed}"],
    )

    # 4) Long rollout split
    # Data partition mirrors ID; shift meaning is evaluation horizon, not case partition.
    split_long = make_split_payload(
        split_name=f"split_{args.dataset_version}_long_rollout",
        dataset_version=args.dataset_version,
        train_ids=id_train,
        val_ids=id_val,
        test_ids=id_test,
        test_shift_ids=id_test,
        shift_definition="Train with short rollout horizon, evaluate with long rollout horizon on test_shift_case_ids.",
        notes=["Case partition mirrors ID split", f"seed={args.seed}"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    id_path = args.output_dir / f"split_{args.dataset_version}_id.json"
    param_path = args.output_dir / f"split_{args.dataset_version}_param_shift.json"
    geo_path = args.output_dir / f"split_{args.dataset_version}_geometry_shift.json"
    long_path = args.output_dir / f"split_{args.dataset_version}_long_rollout.json"

    write_json(id_path, split_id)
    write_json(param_path, split_param)
    write_json(geo_path, split_geo)
    write_json(long_path, split_long)

    print(f"[OK] Wrote: {id_path}")
    print(f"[OK] Wrote: {param_path}")
    print(f"[OK] Wrote: {geo_path}")
    print(f"[OK] Wrote: {long_path}")
    print(
        "[INFO] Split sizes | "
        f"ID train/val/test: {len(id_train)}/{len(id_val)}/{len(id_test)} | "
        f"Param shift test_shift: {len(split_param['test_shift_case_ids'])} | "
        f"Geo shift test_shift: {len(split_geo['test_shift_case_ids'])}"
    )

    by_sz: dict[str, int] = defaultdict(int)
    for cid in id_test:
        by_sz[case_by_id[cid].get("scenario_type", "unknown")] += 1
    small = [f"{s}={n}" for s, n in sorted(by_sz.items()) if n < 5]
    if small:
        print(
            "[WARN] ID test has scenarios with <5 cases: "
            + ", ".join(small)
            + " — increase cases-per-scenario or loosen ratios for stable paper metrics."
        )


if __name__ == "__main__":
    main()
