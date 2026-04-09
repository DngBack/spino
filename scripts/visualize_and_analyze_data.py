#!/usr/bin/env python3
"""
Visualize and analyze SPINO dataset metadata.

Usage example:
  python scripts/visualize_and_analyze_data.py \
      --manifest data/metadata/dataset_manifest.template.json \
      --output-dir outputs/data_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze dataset metadata and generate visualizations."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/dataset_manifest.template.json"),
        help="Path to dataset manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data_analysis"),
        help="Directory to save analysis artifacts.",
    )
    parser.add_argument(
        "--max-parameter-plots",
        type=int,
        default=8,
        help="Maximum number of parameter histogram subplots.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cases(manifest: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in manifest.get("cases", []):
        meta_rel = case.get("metadata_path")
        if not meta_rel:
            continue
        meta_path = repo_root / meta_rel
        if not meta_path.exists():
            row = {
                "case_id": case.get("case_id", "unknown"),
                "metadata_path": str(meta_path),
                "metadata_exists": False,
                "scenario_type": case.get("scenario_type"),
                "geometry_id": case.get("geometry_id"),
                "seed": case.get("seed"),
                "resolution_tag": None,
                "num_steps": None,
                "dt": None,
                "split_tags": ",".join(case.get("split_tags", [])),
            }
            rows.append(row)
            continue

        meta = load_json(meta_path)
        mesh = meta.get("mesh_or_grid", {})
        time_info = meta.get("time", {})
        params = meta.get("parameters", {})
        flags = meta.get("quality_flags", {})
        stimulus = meta.get("stimulus", {})

        row = {
            "case_id": meta.get("case_id", case.get("case_id", "unknown")),
            "metadata_path": str(meta_path),
            "metadata_exists": True,
            "dataset_version": meta.get("dataset_version"),
            "generator": meta.get("generator"),
            "scenario_type": meta.get("scenario_type"),
            "geometry_id": meta.get("geometry_id"),
            "seed": meta.get("seed"),
            "resolution_tag": mesh.get("resolution_tag"),
            "shape_h": (mesh.get("shape") or [None, None])[0],
            "shape_w": (mesh.get("shape") or [None, None])[1],
            "dx": mesh.get("dx"),
            "num_steps": time_info.get("num_steps"),
            "dt": time_info.get("dt"),
            "split_tags": ",".join(meta.get("split_tags", [])),
            "stimulus_type": stimulus.get("type"),
            "stimulus_amplitude": stimulus.get("amplitude"),
            "has_nan": flags.get("has_nan"),
            "has_inf": flags.get("has_inf"),
            "is_duplicate_hash": flags.get("is_duplicate_hash"),
            "physics_warning": flags.get("physics_warning"),
        }

        for key, value in params.items():
            row[f"param_{key}"] = value

        rows.append(row)
    return rows


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_plots(df: pd.DataFrame, output_dir: Path, max_parameter_plots: int) -> list[str]:
    generated: list[str] = []
    viz_dir = output_dir / "figures"
    viz_dir.mkdir(parents=True, exist_ok=True)

    if "scenario_type" in df and not df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x="scenario_type", order=df["scenario_type"].value_counts().index, ax=ax)
        ax.set_title("Case Count by Scenario")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=20)
        out = viz_dir / "scenario_counts.png"
        _save_plot(fig, out)
        generated.append(str(out))

    if "split_tags" in df and not df.empty:
        split_series = (
            df["split_tags"].fillna("").apply(lambda s: s.split(",") if s else ["unassigned"]).explode()
        )
        split_df = split_series.to_frame(name="split_tag")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(
            data=split_df,
            x="split_tag",
            order=split_df["split_tag"].value_counts().index,
            ax=ax,
        )
        ax.set_title("Case Count by Split Tag")
        ax.set_xlabel("Split Tag")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=20)
        out = viz_dir / "split_counts.png"
        _save_plot(fig, out)
        generated.append(str(out))

    if "resolution_tag" in df and not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(
            data=df,
            x="resolution_tag",
            order=df["resolution_tag"].value_counts(dropna=False).index,
            ax=ax,
        )
        ax.set_title("Case Count by Resolution")
        ax.set_xlabel("Resolution")
        ax.set_ylabel("Count")
        out = viz_dir / "resolution_counts.png"
        _save_plot(fig, out)
        generated.append(str(out))

    if not df.empty and {"scenario_type", "split_tags"}.issubset(df.columns):
        expanded = df[["scenario_type", "split_tags"]].copy()
        expanded["split_tags"] = expanded["split_tags"].fillna("").apply(
            lambda s: s.split(",") if s else ["unassigned"]
        )
        expanded = expanded.explode("split_tags")
        pivot = (
            expanded.groupby(["scenario_type", "split_tags"])
            .size()
            .reset_index(name="count")
            .pivot(index="scenario_type", columns="split_tags", values="count")
            .fillna(0)
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", ax=ax)
        ax.set_title("Scenario vs Split Tag")
        ax.set_xlabel("Split Tag")
        ax.set_ylabel("Scenario")
        out = viz_dir / "scenario_split_heatmap.png"
        _save_plot(fig, out)
        generated.append(str(out))

    param_cols = [c for c in df.columns if c.startswith("param_")]
    param_cols = param_cols[:max_parameter_plots]
    if param_cols:
        n_cols = 3
        n_rows = (len(param_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
        if hasattr(axes, "flatten"):
            axes = list(axes.flatten())
        else:
            axes = [axes]
        for i, col in enumerate(param_cols):
            ax = axes[i]
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if vals.empty:
                ax.set_visible(False)
                continue
            sns.histplot(vals, bins=20, kde=True, ax=ax)
            ax.set_title(col.replace("param_", ""))
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
        for j in range(len(param_cols), len(axes)):
            axes[j].set_visible(False)
        out = viz_dir / "parameter_histograms.png"
        _save_plot(fig, out)
        generated.append(str(out))

    return generated


def build_summary(df: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset_name": manifest.get("dataset_name"),
        "dataset_version": manifest.get("dataset_version"),
        "num_manifest_cases": len(manifest.get("cases", [])),
        "num_loaded_cases": int(len(df)),
        "num_missing_metadata": int((~df["metadata_exists"]).sum()) if "metadata_exists" in df else 0,
    }

    if not df.empty:
        if "scenario_type" in df:
            summary["scenario_counts"] = df["scenario_type"].value_counts(dropna=False).to_dict()
        if "resolution_tag" in df:
            summary["resolution_counts"] = df["resolution_tag"].value_counts(dropna=False).to_dict()
        if "generator" in df:
            summary["generator_counts"] = df["generator"].value_counts(dropna=False).to_dict()
        if "geometry_id" in df:
            summary["num_unique_geometries"] = int(df["geometry_id"].nunique(dropna=True))
        if "seed" in df:
            summary["num_unique_seeds"] = int(df["seed"].nunique(dropna=True))
        if "num_steps" in df:
            summary["num_steps_stats"] = {
                "min": float(pd.to_numeric(df["num_steps"], errors="coerce").min()),
                "max": float(pd.to_numeric(df["num_steps"], errors="coerce").max()),
                "mean": float(pd.to_numeric(df["num_steps"], errors="coerce").mean()),
            }
        qc_cols = ["has_nan", "has_inf", "is_duplicate_hash", "physics_warning"]
        summary["quality_flag_counts"] = {}
        for col in qc_cols:
            if col in df:
                summary["quality_flag_counts"][col] = int((df[col] == True).sum())  # noqa: E712

    return summary


def write_markdown_report(summary: dict[str, Any], plots: list[str], output_path: Path) -> None:
    lines = [
        "# SPINO Data Analysis Report",
        "",
        "## Summary",
        f"- Dataset name: `{summary.get('dataset_name')}`",
        f"- Dataset version: `{summary.get('dataset_version')}`",
        f"- Cases in manifest: `{summary.get('num_manifest_cases')}`",
        f"- Cases loaded: `{summary.get('num_loaded_cases')}`",
        f"- Missing metadata files: `{summary.get('num_missing_metadata')}`",
        "",
        "## Counts",
        f"- Scenario counts: `{summary.get('scenario_counts', {})}`",
        f"- Resolution counts: `{summary.get('resolution_counts', {})}`",
        f"- Generator counts: `{summary.get('generator_counts', {})}`",
        f"- Unique geometries: `{summary.get('num_unique_geometries')}`",
        f"- Unique seeds: `{summary.get('num_unique_seeds')}`",
        "",
        "## Quality flags",
        f"- Flag counts: `{summary.get('quality_flag_counts', {})}`",
        "",
        "## Generated figures",
    ]
    for p in plots:
        lines.append(f"- `{p}`")
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(".").resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    manifest = load_json(args.manifest)
    rows = load_cases(manifest, repo_root)
    df = pd.DataFrame(rows)

    table_path = output_dir / "cases_table.csv"
    df.to_csv(table_path, index=False)

    summary = build_summary(df, manifest)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plots = make_plots(df, output_dir, args.max_parameter_plots)
    report_path = output_dir / "report.md"
    write_markdown_report(summary, plots, report_path)

    print(f"[OK] Cases table: {table_path}")
    print(f"[OK] Summary JSON: {summary_path}")
    print(f"[OK] Report: {report_path}")
    if plots:
        print("[OK] Figures:")
        for p in plots:
            print(f"  - {p}")
    else:
        print("[WARN] No figures generated (insufficient data).")


if __name__ == "__main__":
    main()
