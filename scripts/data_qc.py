#!/usr/bin/env python3
"""
Run data quality checks for SPINO dataset.

Checks include:
- manifest integrity,
- metadata completeness,
- split leakage checks,
- optional tensor existence and NaN/Inf checks.

Example:
  python scripts/data_qc.py \
    --manifest data/metadata/dataset_manifest.v1.0.json \
    --check-tensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_METADATA_FIELDS = [
    "case_id",
    "dataset_version",
    "generator",
    "scenario_type",
    "geometry_id",
    "mesh_or_grid",
    "time",
    "seed",
    "parameters",
    "stimulus",
    "state_channels",
    "file_paths",
    "split_tags",
    "quality_flags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QC checks on SPINO dataset metadata/splits.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/dataset_manifest.v1.0.json"),
        help="Path to manifest JSON.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root for resolving relative paths.",
    )
    parser.add_argument(
        "--check-tensors",
        action="store_true",
        help="If set, loads tensor files and checks NaN/Inf + shape consistency.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)


def warn(msg: str, warnings: list[str]) -> None:
    warnings.append(msg)


def infer_case_signature(meta: dict[str, Any], manifest_case: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    scenario = meta.get("scenario_type", manifest_case.get("scenario_type"))
    geometry_id = meta.get("geometry_id", manifest_case.get("geometry_id"))
    parameter_set_id = manifest_case.get("parameter_set_id", "unknown")
    seed = meta.get("seed", manifest_case.get("seed"))
    return scenario, geometry_id, parameter_set_id, seed


def check_tensor(path: Path, expected_steps: int | None, expected_shape: list[int] | None) -> tuple[bool, str]:
    if not path.exists():
        return False, f"tensor file missing: {path}"
    try:
        if path.suffix == ".npz":
            with np.load(path) as data:
                if not data.files:
                    return False, f"npz has no arrays: {path}"
                arr = data[data.files[0]]
        elif path.suffix == ".npy":
            arr = np.load(path)
        else:
            return True, f"skip tensor content check for unsupported extension: {path.suffix}"
    except Exception as exc:
        return False, f"failed to load tensor {path}: {exc}"

    if np.isnan(arr).any():
        return False, f"NaN found in tensor: {path}"
    if np.isinf(arr).any():
        return False, f"Inf found in tensor: {path}"

    if expected_steps is not None and arr.ndim >= 1 and int(arr.shape[0]) != int(expected_steps):
        return False, f"num_steps mismatch tensor={arr.shape[0]} metadata={expected_steps} path={path}"
    if expected_shape and arr.ndim >= 3:
        # For expected shape [H, W], tensor often [T, H, W, C] or [T, C, H, W].
        h, w = expected_shape[0], expected_shape[1]
        if not ((arr.shape[1] == h and arr.shape[2] == w) or (arr.shape[-3] == h and arr.shape[-2] == w)):
            return False, f"spatial shape mismatch tensor_shape={arr.shape} expected_hw={expected_shape} path={path}"

    return True, f"tensor OK: {path}"


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest
    errors: list[str] = []
    warnings: list[str] = []

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    cases = manifest.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest has no cases.")

    seen_case_ids: set[str] = set()
    signature_by_split: dict[str, set[tuple[Any, Any, Any]]] = {
        "train": set(),
        "val": set(),
        "test": set(),
        "test_shift": set(),
    }

    case_ids_in_manifest = set()
    case_id_to_signature: dict[str, tuple[Any, Any, Any]] = {}

    for case in cases:
        case_id = case.get("case_id")
        if not case_id:
            fail("Manifest case missing case_id.", errors)
            continue
        if case_id in seen_case_ids:
            fail(f"Duplicate case_id in manifest: {case_id}", errors)
        seen_case_ids.add(case_id)
        case_ids_in_manifest.add(case_id)

        meta_rel = case.get("metadata_path")
        if not meta_rel:
            fail(f"Case {case_id} missing metadata_path in manifest.", errors)
            continue
        meta_path = (repo_root / meta_rel).resolve()
        if not meta_path.exists():
            fail(f"Metadata file does not exist for case {case_id}: {meta_path}", errors)
            continue

        meta = load_json(meta_path)
        for key in REQUIRED_METADATA_FIELDS:
            if key not in meta:
                fail(f"Case {case_id} missing metadata field: {key}", errors)

        if meta.get("case_id") != case_id:
            fail(f"case_id mismatch manifest={case_id} metadata={meta.get('case_id')}", errors)

        file_paths = meta.get("file_paths", {})
        tensor_rel = file_paths.get("processed_tensor_path")
        if not tensor_rel:
            fail(f"Case {case_id} missing file_paths.processed_tensor_path.", errors)
        else:
            tensor_path = (repo_root / tensor_rel).resolve()
            if not tensor_path.exists():
                fail(f"Processed tensor missing for case {case_id}: {tensor_path}", errors)
            elif args.check_tensors:
                time_info = meta.get("time", {})
                expected_steps = time_info.get("num_steps")
                mesh = meta.get("mesh_or_grid", {})
                expected_shape = mesh.get("shape") if isinstance(mesh.get("shape"), list) else None
                ok, message = check_tensor(tensor_path, expected_steps, expected_shape)
                if not ok:
                    fail(message, errors)
                else:
                    # Keep output concise; success entries are not warnings.
                    pass

        flags = meta.get("quality_flags", {})
        for k in ("has_nan", "has_inf", "is_duplicate_hash"):
            if flags.get(k) is True:
                fail(f"Case {case_id} has quality flag {k}=true", errors)
        if flags.get("physics_warning") is True:
            warn(f"Case {case_id} has physics_warning=true", warnings)

        signature = infer_case_signature(meta, case)
        case_id_to_signature[case_id] = signature

    # Split-file checks
    split_files = manifest.get("split_files", {})
    for split_key in ("id", "parameter_shift", "geometry_shift", "long_rollout"):
        split_rel = split_files.get(split_key)
        if not split_rel:
            warn(f"Manifest missing split_files.{split_key}", warnings)
            continue
        split_path = (repo_root / split_rel).resolve()
        if not split_path.exists():
            warn(f"Split file not found: {split_path}", warnings)
            continue
        split = load_json(split_path)
        train_ids = set(split.get("train_case_ids", []))
        val_ids = set(split.get("val_case_ids", []))
        test_ids = set(split.get("test_case_ids", []))
        test_shift_ids = set(split.get("test_shift_case_ids", []))

        unknown = (train_ids | val_ids | test_ids | test_shift_ids) - case_ids_in_manifest
        if unknown:
            fail(f"Split {split_key} contains unknown case_ids: {sorted(list(unknown))[:10]}", errors)

        if train_ids & val_ids:
            fail(f"Split {split_key} leakage train/val overlap.", errors)
        if train_ids & test_ids:
            fail(f"Split {split_key} leakage train/test overlap.", errors)
        if train_ids & test_shift_ids:
            fail(f"Split {split_key} leakage train/test_shift overlap.", errors)
        if val_ids & test_ids:
            fail(f"Split {split_key} leakage val/test overlap.", errors)
        if val_ids & test_shift_ids:
            fail(f"Split {split_key} leakage val/test_shift overlap.", errors)

        # Signature leakage check for train vs test/test_shift.
        train_signatures = {case_id_to_signature[c] for c in train_ids if c in case_id_to_signature}
        test_signatures = {case_id_to_signature[c] for c in test_ids if c in case_id_to_signature}
        test_shift_signatures = {case_id_to_signature[c] for c in test_shift_ids if c in case_id_to_signature}
        if train_signatures & test_signatures:
            fail(f"Split {split_key} signature leakage train/test.", errors)
        if train_signatures & test_shift_signatures:
            fail(f"Split {split_key} signature leakage train/test_shift.", errors)

        signature_by_split["train"].update(train_signatures)
        signature_by_split["val"].update({case_id_to_signature[c] for c in val_ids if c in case_id_to_signature})
        signature_by_split["test"].update(test_signatures)
        signature_by_split["test_shift"].update(test_shift_signatures)

    # Summary
    print("=== DATA QC SUMMARY ===")
    print(f"Manifest: {manifest_path}")
    print(f"Total cases: {len(case_ids_in_manifest)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    if warnings:
        print("\n--- WARNINGS ---")
        for w in warnings[:50]:
            print(f"[WARN] {w}")
        if len(warnings) > 50:
            print(f"[WARN] ... ({len(warnings) - 50} more)")
    if errors:
        print("\n--- ERRORS ---")
        for e in errors[:100]:
            print(f"[ERR] {e}")
        if len(errors) > 100:
            print(f"[ERR] ... ({len(errors) - 100} more)")
        raise SystemExit(1)
    print("[OK] QC passed.")


if __name__ == "__main__":
    main()
