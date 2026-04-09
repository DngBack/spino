#!/usr/bin/env python3
"""
Build dataset manifest from case metadata files.

Example:
  python scripts/build_manifest.py \
    --metadata-dir data/metadata/case_metadata \
    --output-manifest data/metadata/dataset_manifest.v1.0.json \
    --dataset-name spino_ep2d \
    --dataset-version v1.0 \
    --generation-config-path data/metadata/generation_config.v1.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset manifest from case metadata JSON files.")
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data/metadata/case_metadata"),
        help="Directory containing one metadata JSON per case.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/metadata/dataset_manifest.v1.0.json"),
        help="Output manifest file path.",
    )
    parser.add_argument("--dataset-name", type=str, default="spino_ep2d", help="Dataset name.")
    parser.add_argument("--dataset-version", type=str, default="v1.0", help="Dataset version.")
    parser.add_argument(
        "--generation-config-path",
        type=str,
        default="data/metadata/generation_config.v1.json",
        help="Path recorded in manifest for generation config.",
    )
    parser.add_argument(
        "--split-id-path",
        type=str,
        default="data/splits/split_v1_id.json",
        help="ID split path recorded in manifest.",
    )
    parser.add_argument(
        "--split-parameter-path",
        type=str,
        default="data/splits/split_v1_param_shift.json",
        help="Parameter shift split path recorded in manifest.",
    )
    parser.add_argument(
        "--split-geometry-path",
        type=str,
        default="data/splits/split_v1_geometry_shift.json",
        help="Geometry shift split path recorded in manifest.",
    )
    parser.add_argument(
        "--split-long-rollout-path",
        type=str,
        default="data/splits/split_v1_long_rollout.json",
        help="Long rollout split path recorded in manifest.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def extract_parameter_set_id(case_id: str) -> str:
    # Expected style includes `_p03_`; fallback to "unknown".
    match = re.search(r"_p([A-Za-z0-9]+)_", case_id)
    if not match:
        return "unknown"
    return f"p{match.group(1)}"


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(".").resolve()
    metadata_dir = args.metadata_dir
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    meta_files = sorted(p for p in metadata_dir.glob("*.json") if p.name != "case_metadata.template.json")
    cases: list[dict[str, Any]] = []
    scenario_counter: Counter[str] = Counter()
    resolution_counter: Counter[str] = Counter()
    total_steps = 0
    channels_input: list[str] = []
    channels_output: list[str] = []

    for meta_path in meta_files:
        meta = load_json(meta_path)
        case_id = meta.get("case_id")
        if not case_id:
            raise ValueError(f"Missing 'case_id' in metadata: {meta_path}")

        file_paths = meta.get("file_paths", {})
        tensor_path = file_paths.get("processed_tensor_path")
        if not tensor_path:
            raise ValueError(f"Missing 'file_paths.processed_tensor_path' in metadata: {meta_path}")

        tensor_rel = str(tensor_path)
        meta_rel = relative_to_repo(meta_path, repo_root)
        scenario = meta.get("scenario_type", "unknown")
        geometry_id = meta.get("geometry_id", "unknown")
        seed = meta.get("seed", -1)
        split_tags = meta.get("split_tags", [])
        parameter_set_id = meta.get("parameter_set_id") or extract_parameter_set_id(case_id)

        mesh = meta.get("mesh_or_grid", {})
        resolution_tag = mesh.get("resolution_tag", "unknown")
        time_info = meta.get("time", {})
        num_steps = int(time_info.get("num_steps", 0) or 0)
        total_steps += max(num_steps, 0)

        state_channels = meta.get("state_channels", {})
        if not channels_input and isinstance(state_channels.get("input_channels"), list):
            channels_input = state_channels["input_channels"]
        if not channels_output and isinstance(state_channels.get("output_channels"), list):
            channels_output = state_channels["output_channels"]

        scenario_counter[scenario] += 1
        resolution_counter[resolution_tag] += 1

        cases.append(
            {
                "case_id": case_id,
                "metadata_path": meta_rel,
                "processed_tensor_path": tensor_rel,
                "scenario_type": scenario,
                "geometry_id": geometry_id,
                "parameter_set_id": parameter_set_id,
                "seed": seed,
                "split_tags": split_tags if isinstance(split_tags, list) else [],
            }
        )

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_version": args.dataset_version,
        "created_at": now_iso(),
        "description": "SPINO 2D cardiac electrophysiology benchmark dataset",
        "scenarios": sorted(scenario_counter.keys()),
        "resolutions": sorted(resolution_counter.keys()),
        "num_total_cases": len(cases),
        "num_total_trajectories": len(cases),
        "num_total_steps": total_steps,
        "channels": {
            "input": channels_input,
            "output": channels_output,
        },
        "cases": cases,
        "split_files": {
            "id": args.split_id_path,
            "parameter_shift": args.split_parameter_path,
            "geometry_shift": args.split_geometry_path,
            "long_rollout": args.split_long_rollout_path,
        },
        "generation_config_path": args.generation_config_path,
        "notes": [
            "Auto-generated by scripts/build_manifest.py",
            "All listed paths should be relative to repository root",
        ],
    }
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Wrote manifest: {args.output_manifest}")
    print(f"[OK] Cases: {manifest['num_total_cases']}")
    print(f"[OK] Scenarios: {manifest['scenarios']}")
    print(f"[OK] Resolutions: {manifest['resolutions']}")


if __name__ == "__main__":
    main()
