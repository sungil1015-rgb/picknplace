from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def current_rss_mb() -> float | None:
    status_path = Path("/proc/self/status")
    if not status_path.is_file():
        return None
    for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1]) / 1024.0
    return None


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_records(root: Path, pattern: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob(pattern)):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            print(f"skip {path}: {exc}")
            continue
        records.append({"path": path, "data": data})
    return records


def build_timing_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        data = rec["data"]
        timing = data.get("timing") or {}
        row = {
            "file": rec["path"].name,
            "sample": data.get("sample"),
            "state": data.get("state"),
            "object_count": len(data.get("objects") or []),
        }
        memory_summary = data.get("memory_summary") or {}
        for key in [
            "peak_rss_mb",
            "max_rss_mb",
            "max_cuda_allocated_mb",
            "max_cuda_reserved_mb",
            "max_cuda_peak_allocated_mb",
            "max_cuda_peak_reserved_mb",
        ]:
            row[key] = safe_float(memory_summary.get(key))
        row["cuda_available"] = bool(memory_summary.get("cuda_available"))
        row["cuda_device_name"] = memory_summary.get("cuda_device_name")
        for key in ["seg_sec", "classifi_sec", "normal_sec", "prior_sec", "run_sec", "debug_suction_sec"]:
            row[key] = safe_float(timing.get(key))
        debug_suction = 0.0 if np.isnan(row["debug_suction_sec"]) else row["debug_suction_sec"]
        row["run_plus_debug_sec"] = row["run_sec"] + debug_suction
        rows.append(row)
    return pd.DataFrame(rows)


def build_inference_memory_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        data = rec["data"]
        for snapshot in data.get("memory") or []:
            row = {
                "file": rec["path"].name,
                "sample": data.get("sample"),
                "stage": snapshot.get("stage"),
                "elapsed_sec": safe_float(snapshot.get("elapsed_sec")),
                "rss_mb": safe_float(snapshot.get("rss_mb")),
                "peak_rss_mb": safe_float(snapshot.get("peak_rss_mb")),
                "cuda_available": bool(snapshot.get("cuda_available")),
                "cuda_device_index": snapshot.get("cuda_device_index"),
                "cuda_device_name": snapshot.get("cuda_device_name"),
                "cuda_allocated_mb": safe_float(snapshot.get("cuda_allocated_mb")),
                "cuda_reserved_mb": safe_float(snapshot.get("cuda_reserved_mb")),
                "cuda_max_allocated_mb": safe_float(snapshot.get("cuda_max_allocated_mb")),
                "cuda_max_reserved_mb": safe_float(snapshot.get("cuda_max_reserved_mb")),
                "reason": snapshot.get("reason"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def priority_group(row: pd.Series) -> str:
    if bool(row.get("unknown_estimated")):
        return "unknown_estimated_low_sim_low_vote"
    if bool(row.get("low_similarity_only_excluded")):
        return "low_sim_only_excluded"
    if bool(row.get("known_priority_candidate")):
        return "known_candidate"
    if bool(row.get("low_vote")):
        return "low_vote_other"
    if bool(row.get("low_similarity")):
        return "low_sim_other"
    return "other"


def build_object_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        data = rec["data"]
        timing = data.get("timing") or {}
        for obj in data.get("objects") or []:
            cls = obj.get("classification") or {}
            pr = obj.get("priority") or {}
            stage = pr.get("priority_stage") or {}
            footprint = obj.get("footprint") or {}
            surface = pr.get("suction_surface") or {}
            rows.append(
                {
                    "file": rec["path"].name,
                    "sample": data.get("sample"),
                    "rank": obj.get("rank"),
                    "object_number": obj.get("object_number"),
                    "is_grasp_target": bool(obj.get("is_grasp_target")),
                    "is_suction_debug_target": bool(obj.get("is_suction_debug_target")),
                    "class_id_result": obj.get("class_id"),
                    "class_index": cls.get("class_index"),
                    "class_name": cls.get("class_name"),
                    "reject_reason": cls.get("reject_reason"),
                    "confidence": safe_float(cls.get("confidence")),
                    "similarity": safe_float(cls.get("similarity")),
                    "vote_ratio": safe_float(cls.get("vote_ratio")),
                    "margin": safe_float(cls.get("margin")),
                    "detector_score": safe_float(cls.get("detector_score")),
                    "neighbor_labels": tuple(cls.get("neighbor_labels") or []),
                    "neighbor_similarities": tuple(cls.get("neighbor_similarities") or []),
                    "priority": safe_float(pr.get("priority")),
                    "grasp_depth": safe_float(pr.get("grasp_depth")),
                    "grasp_depth_source": pr.get("grasp_depth_source"),
                    "priority_depth_source": pr.get("priority_depth_source"),
                    "class_similarity_priority": safe_float(pr.get("class_similarity")),
                    "class_vote_ratio_priority": safe_float(pr.get("class_vote_ratio")),
                    "class_reject_reason_priority": pr.get("class_reject_reason"),
                    "valid_depth": bool(pr.get("valid_depth")),
                    "mask_area": safe_float(pr.get("mask_area")),
                    "suction_normal_z_score": safe_float(pr.get("suction_normal_z_score")),
                    "footprint_feasible": footprint.get("feasible"),
                    "footprint_reason": footprint.get("reason"),
                    "surface_passed": surface.get("passed"),
                    "surface_reason": surface.get("reason"),
                    "surface_area": safe_float(surface.get("surface_area")),
                    "surface_area_ratio": safe_float(surface.get("surface_area_ratio")),
                    "normal_angular_std_deg": safe_float(surface.get("normal_angular_std_deg")),
                    "robot_z_tilt_deg": safe_float(surface.get("robot_z_tilt_deg")),
                    "depth_rank": stage.get("depth_rank"),
                    "valid_depth_count": stage.get("valid_depth_count"),
                    "low_similarity": bool(stage.get("low_similarity")),
                    "low_vote": bool(stage.get("low_vote")),
                    "known_priority_candidate": bool(stage.get("known_priority_candidate")),
                    "unknown_estimated": bool(stage.get("unknown_estimated")),
                    "low_similarity_only_excluded": bool(stage.get("low_similarity_only_excluded")),
                    "priority_excluded_reason": stage.get("priority_excluded_reason"),
                    "in_known_pool": bool(stage.get("in_known_pool")),
                    "in_unknown_estimated_pool": bool(stage.get("in_unknown_estimated_pool")),
                    "in_final_pool": bool(stage.get("in_final_pool")),
                    "final_selected": bool(stage.get("final_selected")),
                    "seg_sec": safe_float(timing.get("seg_sec")),
                    "classifi_sec": safe_float(timing.get("classifi_sec")),
                    "normal_sec": safe_float(timing.get("normal_sec")),
                    "prior_sec": safe_float(timing.get("prior_sec")),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["priority_group"] = df.apply(priority_group, axis=1)
        df["class_label"] = df["class_index"].astype(str) + ":" + df["class_name"].astype(str)
    return df


def save_timing_reports(timing_df: pd.DataFrame, output_dir: Path) -> None:
    timing_cols = ["seg_sec", "classifi_sec", "normal_sec", "prior_sec", "run_sec", "debug_suction_sec", "run_plus_debug_sec"]
    timing_summary = timing_df[timing_cols].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).T
    timing_summary["sum_sec"] = timing_df[timing_cols].sum(numeric_only=True)
    timing_summary.to_csv(output_dir / "timing_summary.csv")
    timing_df.sort_values("run_plus_debug_sec", ascending=False).head(20).to_csv(output_dir / "slow_cases_top20.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    stage_cols = ["seg_sec", "classifi_sec", "normal_sec", "prior_sec"]
    timing_df[stage_cols].mean().sort_values(ascending=False).plot(kind="bar", ax=axes[0], color="#4C78A8")
    axes[0].set_title("Average Time by Stage")
    axes[0].set_ylabel("seconds")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].boxplot([timing_df[col].dropna().values for col in stage_cols], tick_labels=stage_cols, showfliers=False)
    axes[1].set_title("Time Distribution by Stage")
    axes[1].set_ylabel("seconds")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "timing_stage_summary.png", dpi=160)
    plt.close(fig)


def save_inference_memory_reports(memory_df: pd.DataFrame, timing_df: pd.DataFrame, output_dir: Path) -> None:
    if memory_df.empty:
        pd.DataFrame(
            [
                {
                    "note": "No inference memory data found. Re-run src/utils/debug_local.py after memory logging patch.",
                }
            ]
        ).to_csv(output_dir / "inference_memory_usage.csv", index=False)
        return

    memory_df.to_csv(output_dir / "inference_memory_usage.csv", index=False)
    overall_rows: list[dict[str, Any]] = []
    for col in [
        "rss_mb",
        "peak_rss_mb",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
        "cuda_max_allocated_mb",
        "cuda_max_reserved_mb",
    ]:
        values = memory_df[col].dropna() if col in memory_df.columns else pd.Series(dtype=float)
        overall_rows.append(
            {
                "metric": col,
                "count": int(values.count()),
                "mean_mb": float(values.mean()) if len(values) else float("nan"),
                "p95_mb": float(values.quantile(0.95)) if len(values) else float("nan"),
                "max_mb": float(values.max()) if len(values) else float("nan"),
            }
        )
    pd.DataFrame(overall_rows).to_csv(output_dir / "memory_overall_summary.csv", index=False)

    memory_summary = (
        memory_df.groupby("stage", dropna=False)
        .agg(
            count=("file", "size"),
            rss_mean_mb=("rss_mb", "mean"),
            rss_p95_mb=("rss_mb", lambda s: s.quantile(0.95)),
            rss_max_mb=("rss_mb", "max"),
            peak_rss_max_mb=("peak_rss_mb", "max"),
            cuda_allocated_mean_mb=("cuda_allocated_mb", "mean"),
            cuda_allocated_p95_mb=("cuda_allocated_mb", lambda s: s.quantile(0.95)),
            cuda_allocated_max_mb=("cuda_allocated_mb", "max"),
            cuda_reserved_max_mb=("cuda_reserved_mb", "max"),
            cuda_peak_allocated_max_mb=("cuda_max_allocated_mb", "max"),
            cuda_peak_reserved_max_mb=("cuda_max_reserved_mb", "max"),
        )
        .reset_index()
    )
    memory_summary.to_csv(output_dir / "inference_memory_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    stages = list(memory_summary["stage"].astype(str))
    axes[0].plot(stages, memory_summary["rss_mean_mb"], marker="o", label="rss_mean_mb")
    axes[0].plot(stages, memory_summary["rss_p95_mb"], marker="o", label="rss_p95_mb")
    axes[0].plot(stages, memory_summary["rss_max_mb"], marker="o", label="rss_max_mb")
    axes[0].set_title("CPU RSS by Debug Stage")
    axes[0].set_ylabel("MB")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=8)

    if memory_df["cuda_available"].any():
        axes[1].plot(stages, memory_summary["cuda_allocated_p95_mb"], marker="o", label="cuda_allocated_p95_mb")
        axes[1].plot(stages, memory_summary["cuda_allocated_max_mb"], marker="o", label="cuda_allocated_max_mb")
        axes[1].plot(stages, memory_summary["cuda_reserved_max_mb"], marker="o", label="cuda_reserved_max_mb")
        axes[1].set_title("CUDA Memory by Debug Stage")
        axes[1].set_ylabel("MB")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "CUDA memory not available in JSON", ha="center", va="center")
        axes[1].set_axis_off()
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "inference_memory_by_stage.png", dpi=160)
    plt.close(fig)

    available_cols = [col for col in ["peak_rss_mb", "max_cuda_peak_allocated_mb", "max_cuda_peak_reserved_mb"] if col in timing_df.columns]
    if available_cols:
        fig, axes = plt.subplots(1, len(available_cols), figsize=(6 * len(available_cols), 5))
        axes_arr = np.atleast_1d(axes)
        for ax, col in zip(axes_arr, available_cols):
            ax.scatter(timing_df["object_count"], timing_df[col], alpha=0.7)
            ax.set_title(f"{col} vs object_count")
            ax.set_xlabel("object_count")
            ax.set_ylabel("MB")
        fig.tight_layout()
        fig.savefig(output_dir / "inference_memory_vs_object_count.png", dpi=160)
        plt.close(fig)

    save_memory_one_page_summary(memory_df, output_dir)


def save_memory_one_page_summary(memory_df: pd.DataFrame, output_dir: Path) -> None:
    if memory_df.empty:
        return

    metrics = [
        ("CPU RAM\\ncurrent avg", "rss_mb", "mean", "#4C78A8"),
        ("CPU RAM\\npeak", "peak_rss_mb", "max", "#72B7B2"),
        ("GPU actual\\ncurrent avg", "cuda_allocated_mb", "mean", "#F58518"),
        ("GPU actual\\npeak", "cuda_max_allocated_mb", "max", "#E45756"),
        ("GPU reserved\\npeak", "cuda_max_reserved_mb", "max", "#B279A2"),
    ]

    rows: list[dict[str, Any]] = []
    for label, col, agg, color in metrics:
        values = memory_df[col].dropna() if col in memory_df.columns else pd.Series(dtype=float)
        if len(values) == 0:
            value = float("nan")
        elif agg == "mean":
            value = float(values.mean())
        else:
            value = float(values.max())
        rows.append({"label": label, "column": col, "agg": agg, "value_mb": value, "color": color})

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "memory_one_page_summary.csv", index=False)
    plot_df = summary.dropna(subset=["value_mb"]).copy()
    if plot_df.empty:
        return

    fig = plt.figure(figsize=(13, 7), facecolor="white", constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.22)
    card_grid = grid[0].subgridspec(1, len(plot_df), wspace=0.12)
    for index, row in enumerate(plot_df.itertuples(index=False)):
        ax = fig.add_subplot(card_grid[0, index])
        ax.set_facecolor("#F7F7F7")
        for spine in ax.spines.values():
            spine.set_color("#DDDDDD")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.68, f"{row.value_mb / 1024.0:.2f} GB", ha="center", va="center", fontsize=21, fontweight="bold", color=row.color)
        ax.text(0.5, 0.35, row.label, ha="center", va="center", fontsize=11, color="#333333")
        ax.text(0.5, 0.15, f"{row.value_mb:.0f} MB", ha="center", va="center", fontsize=9, color="#666666")

    ax_bar = fig.add_subplot(grid[1])
    bars = ax_bar.bar(plot_df["label"], plot_df["value_mb"] / 1024.0, color=plot_df["color"])
    ax_bar.set_title("Memory Summary", fontsize=15, fontweight="bold")
    ax_bar.set_ylabel("GB")
    ax_bar.tick_params(axis="x", labelrotation=0)
    ax_bar.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, plot_df["value_mb"] / 1024.0):
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.2f} GB", ha="center", va="bottom", fontsize=10)

    cpu_peak = float(plot_df.loc[plot_df["column"] == "peak_rss_mb", "value_mb"].max()) if (plot_df["column"] == "peak_rss_mb").any() else float("nan")
    gpu_peak = float(plot_df.loc[plot_df["column"] == "cuda_max_allocated_mb", "value_mb"].max()) if (plot_df["column"] == "cuda_max_allocated_mb").any() else float("nan")
    subtitle = []
    if np.isfinite(cpu_peak):
        subtitle.append(f"CPU peak {cpu_peak / 1024.0:.2f} GB")
    if np.isfinite(gpu_peak):
        subtitle.append(f"GPU actual peak {gpu_peak / 1024.0:.2f} GB")
    if subtitle:
        fig.suptitle(" / ".join(subtitle), fontsize=12, y=0.98, color="#444444")

    fig.savefig(output_dir / "memory_one_page_summary.png", dpi=180)
    plt.close(fig)


def save_class_reports(obj_df: pd.DataFrame, output_dir: Path) -> None:
    class_summary = (
        obj_df.groupby(["class_index", "class_name"], dropna=False)
        .agg(
            count=("file", "size"),
            sim_mean=("similarity", "mean"),
            sim_std=("similarity", "std"),
            sim_min=("similarity", "min"),
            sim_p05=("similarity", lambda s: s.quantile(0.05)),
            sim_median=("similarity", "median"),
            sim_p95=("similarity", lambda s: s.quantile(0.95)),
            vote_mean=("vote_ratio", "mean"),
            vote_min=("vote_ratio", "min"),
            low_sim_rate=("low_similarity", "mean"),
            low_vote_rate=("low_vote", "mean"),
            unknown_estimated_rate=("unknown_estimated", "mean"),
            grasp_selected_count=("is_grasp_target", "sum"),
        )
        .reset_index()
    )
    class_summary.to_csv(output_dir / "class_similarity_summary.csv", index=False)

    plot_df = obj_df.dropna(subset=["similarity"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_df.boxplot(column="similarity", by="class_label", ax=axes[0], rot=45, showfliers=False)
    axes[0].set_title("Similarity by Predicted Class")
    axes[0].set_xlabel("class")
    axes[0].set_ylabel("similarity")
    plot_df.boxplot(column="vote_ratio", by="class_label", ax=axes[1], rot=45, showfliers=False)
    axes[1].set_title("Vote Ratio by Predicted Class")
    axes[1].set_xlabel("class")
    axes[1].set_ylabel("vote_ratio")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output_dir / "class_similarity_vote_boxplot.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, group in plot_df.groupby("class_label"):
        values = group["similarity"].dropna().values
        if len(values):
            ax.hist(values, bins=35, density=True, histtype="step", linewidth=2, label=label)
    ax.set_title("Similarity Density by Class")
    ax.set_xlabel("similarity")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "class_similarity_density.png", dpi=160)
    plt.close(fig)


def save_priority_reports(obj_df: pd.DataFrame, output_dir: Path) -> None:
    unknown_summary = (
        obj_df.groupby("priority_group")
        .agg(
            count=("file", "size"),
            sim_mean=("similarity", "mean"),
            sim_median=("similarity", "median"),
            vote_mean=("vote_ratio", "mean"),
            vote_median=("vote_ratio", "median"),
            final_selected_count=("final_selected", "sum"),
            grasp_target_count=("is_grasp_target", "sum"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    unknown_summary.to_csv(output_dir / "unknown_priority_group_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    max_area = max(float(obj_df["mask_area"].max()), 1.0)
    for name, group in obj_df.groupby("priority_group"):
        sizes = 25 + 155 * (group["mask_area"].fillna(0).clip(lower=1) / max_area)
        ax.scatter(group["similarity"], group["vote_ratio"], s=sizes, alpha=0.7, label=name)
    targets = obj_df[obj_df["is_grasp_target"]]
    ax.scatter(targets["similarity"], targets["vote_ratio"], s=220, facecolors="none", edgecolors="black", linewidths=1.5, label="grasp_target")
    ax.set_title("Similarity vs Vote Ratio by Priority Group")
    ax.set_xlabel("similarity")
    ax.set_ylabel("vote_ratio")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "unknown_similarity_vote_scatter.png", dpi=180)
    plt.close(fig)

    reject_counts = obj_df["reject_reason"].fillna("pass").value_counts().rename_axis("reject_reason").reset_index(name="count")
    reject_counts.to_csv(output_dir / "reject_reason_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(reject_counts["reject_reason"], reject_counts["count"], color="#F58518")
    ax.invert_yaxis()
    ax.set_title("Reject Reason Counts")
    ax.set_xlabel("count")
    fig.tight_layout()
    fig.savefig(output_dir / "reject_reason_counts.png", dpi=160)
    plt.close(fig)

    stage_cols = [
        "valid_depth",
        "known_priority_candidate",
        "unknown_estimated",
        "low_similarity_only_excluded",
        "in_known_pool",
        "in_unknown_estimated_pool",
        "in_final_pool",
        "final_selected",
        "is_grasp_target",
    ]
    stage_summary = pd.DataFrame(
        {
            "stage": stage_cols,
            "count": [int(obj_df[col].sum()) for col in stage_cols],
            "rate": [float(obj_df[col].mean()) for col in stage_cols],
        }
    )
    stage_summary.to_csv(output_dir / "priority_stage_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(stage_summary["stage"], stage_summary["count"], color="#54A24B")
    ax.set_title("Priority Stage Counts")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "priority_stage_counts.png", dpi=160)
    plt.close(fig)


def save_target_neighbor_surface_reports(obj_df: pd.DataFrame, output_dir: Path) -> None:
    target_df = obj_df[obj_df["is_grasp_target"]].copy()
    target_summary = (
        target_df.groupby(["class_index", "class_name", "priority_group"], dropna=False)
        .agg(
            count=("file", "size"),
            sim_mean=("similarity", "mean"),
            vote_mean=("vote_ratio", "mean"),
            depth_mean=("grasp_depth", "mean"),
            depth_min=("grasp_depth", "min"),
            depth_max=("grasp_depth", "max"),
        )
        .reset_index()
    )
    target_summary.to_csv(output_dir / "grasp_target_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, group in target_df.groupby("priority_group"):
        axes[0].hist(group["similarity"].dropna(), bins=25, alpha=0.6, label=name)
        axes[1].hist(group["grasp_depth"].dropna(), bins=25, alpha=0.6, label=name)
    axes[0].set_title("Selected Target Similarity")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Selected Target Depth")
    axes[1].set_xlabel("depth mm")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "grasp_target_similarity_depth.png", dpi=160)
    plt.close(fig)

    neighbor_rows: list[dict[str, Any]] = []
    for _, row in obj_df.iterrows():
        labels = row.get("neighbor_labels") or []
        sims = row.get("neighbor_similarities") or []
        for idx, label in enumerate(labels):
            neighbor_rows.append(
                {
                    "priority_group": row["priority_group"],
                    "neighbor_rank": idx + 1,
                    "neighbor_label": label,
                    "neighbor_similarity": safe_float(sims[idx] if idx < len(sims) else np.nan),
                }
            )
    neighbor_df = pd.DataFrame(neighbor_rows)
    neighbor_df.to_csv(output_dir / "neighbor_labels.csv", index=False)
    top_neighbor_mix = neighbor_df[neighbor_df["neighbor_rank"] <= 5].groupby(["priority_group", "neighbor_label"]).size().reset_index(name="count")
    pivot = top_neighbor_mix.pivot_table(index="neighbor_label", columns="priority_group", values="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            ax.text(x, y, str(int(pivot.iloc[y, x])), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="count")
    ax.set_title("Top-5 Neighbor Label Counts")
    fig.tight_layout()
    fig.savefig(output_dir / "neighbor_label_heatmap.png", dpi=160)
    plt.close(fig)

    surface_summary = (
        obj_df.groupby(["class_index", "class_name"], dropna=False)
        .agg(
            count=("file", "size"),
            passed_rate=("surface_passed", lambda s: pd.Series(s).fillna(False).mean()),
            area_ratio_median=("surface_area_ratio", "median"),
            normal_std_median=("normal_angular_std_deg", "median"),
            tilt_median=("robot_z_tilt_deg", "median"),
            z_score_median=("suction_normal_z_score", "median"),
        )
        .reset_index()
    )
    surface_summary.to_csv(output_dir / "surface_normal_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, group in obj_df.groupby("class_name"):
        axes[0].scatter(group["surface_area_ratio"], group["normal_angular_std_deg"], alpha=0.7, label=str(name))
        axes[1].scatter(group["robot_z_tilt_deg"], group["suction_normal_z_score"], alpha=0.7, label=str(name))
    axes[0].set_title("Surface Area Ratio vs Normal Std")
    axes[0].set_xlabel("surface_area_ratio")
    axes[0].set_ylabel("normal_angular_std_deg")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Robot Z Tilt vs Normal Z Score")
    axes[1].set_xlabel("robot_z_tilt_deg")
    axes[1].set_ylabel("suction_normal_z_score")
    box_groups = []
    box_labels = []
    for name, group in obj_df.groupby("class_name"):
        values = group["normal_angular_std_deg"].dropna().values
        if len(values):
            box_groups.append(values)
            box_labels.append(str(name))
    axes[2].boxplot(box_groups, tick_labels=box_labels, showfliers=False)
    axes[2].set_title("Normal Angular Std by Class")
    axes[2].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "surface_normal_quality.png", dpi=160)
    plt.close(fig)


def _rate(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.fillna(False).mean())


def save_operational_reports(
    timing_df: pd.DataFrame,
    obj_df: pd.DataFrame,
    output_dir: Path,
    memory_rows: list[dict[str, Any]],
    wall_sec: float,
) -> None:
    total_files = int(len(timing_df))
    total_objects = int(len(obj_df))
    total_run_sec = float(timing_df["run_sec"].sum())
    total_run_plus_debug_sec = float(timing_df["run_plus_debug_sec"].sum())
    target_df = obj_df[obj_df["is_grasp_target"]].copy()

    summary_rows = [
        {"group": "dataset", "metric": "files", "value": total_files},
        {"group": "dataset", "metric": "objects", "value": total_objects},
        {"group": "dataset", "metric": "avg_objects_per_file", "value": float(timing_df["object_count"].mean())},
        {"group": "dataset", "metric": "empty_result_file_rate", "value": float((timing_df["object_count"] == 0).mean())},
        {"group": "latency", "metric": "run_mean_sec", "value": float(timing_df["run_sec"].mean())},
        {"group": "latency", "metric": "run_p95_sec", "value": float(timing_df["run_sec"].quantile(0.95))},
        {"group": "latency", "metric": "run_max_sec", "value": float(timing_df["run_sec"].max())},
        {"group": "latency", "metric": "run_plus_debug_p95_sec", "value": float(timing_df["run_plus_debug_sec"].quantile(0.95))},
        {"group": "latency", "metric": "seg_p95_sec", "value": float(timing_df["seg_sec"].quantile(0.95))},
        {"group": "latency", "metric": "classifi_p95_sec", "value": float(timing_df["classifi_sec"].quantile(0.95))},
        {"group": "latency", "metric": "normal_p95_sec", "value": float(timing_df["normal_sec"].quantile(0.95))},
        {"group": "latency", "metric": "prior_p95_sec", "value": float(timing_df["prior_sec"].quantile(0.95))},
        {"group": "throughput", "metric": "files_per_run_sec", "value": float(total_files / total_run_sec) if total_run_sec > 0 else float("nan")},
        {"group": "throughput", "metric": "objects_per_run_sec", "value": float(total_objects / total_run_sec) if total_run_sec > 0 else float("nan")},
        {"group": "throughput", "metric": "files_per_run_plus_debug_sec", "value": float(total_files / total_run_plus_debug_sec) if total_run_plus_debug_sec > 0 else float("nan")},
        {"group": "quality", "metric": "classification_reject_rate", "value": float(obj_df["reject_reason"].notna().mean())},
        {"group": "quality", "metric": "low_similarity_rate", "value": _rate(obj_df["low_similarity"])},
        {"group": "quality", "metric": "low_vote_rate", "value": _rate(obj_df["low_vote"])},
        {"group": "quality", "metric": "unknown_estimated_rate", "value": _rate(obj_df["unknown_estimated"])},
        {"group": "quality", "metric": "low_similarity_only_excluded_rate", "value": _rate(obj_df["low_similarity_only_excluded"])},
        {"group": "quality", "metric": "valid_depth_rate", "value": _rate(obj_df["valid_depth"])},
        {"group": "quality", "metric": "surface_passed_rate", "value": _rate(obj_df["surface_passed"])},
        {"group": "quality", "metric": "footprint_feasible_rate", "value": _rate(obj_df["footprint_feasible"])},
        {"group": "target", "metric": "target_count", "value": int(len(target_df))},
        {"group": "target", "metric": "target_unknown_estimated_rate", "value": _rate(target_df["unknown_estimated"]) if len(target_df) else float("nan")},
        {"group": "target", "metric": "target_similarity_median", "value": float(target_df["similarity"].median()) if len(target_df) else float("nan")},
        {"group": "target", "metric": "target_depth_median_mm", "value": float(target_df["grasp_depth"].median()) if len(target_df) else float("nan")},
        {"group": "analysis_script", "metric": "wall_sec", "value": float(wall_sec)},
        {"group": "analysis_script", "metric": "peak_rss_mb", "value": peak_rss_mb()},
    ]
    if "peak_rss_mb" in timing_df.columns and timing_df["peak_rss_mb"].notna().any():
        summary_rows.extend(
            [
                {"group": "inference_memory", "metric": "peak_rss_max_mb", "value": float(timing_df["peak_rss_mb"].max())},
                {"group": "inference_memory", "metric": "rss_max_mb", "value": float(timing_df["max_rss_mb"].max())},
            ]
        )
    if "max_cuda_peak_allocated_mb" in timing_df.columns and timing_df["max_cuda_peak_allocated_mb"].notna().any():
        summary_rows.extend(
            [
                {
                    "group": "inference_memory",
                    "metric": "cuda_peak_allocated_max_mb",
                    "value": float(timing_df["max_cuda_peak_allocated_mb"].max()),
                },
                {
                    "group": "inference_memory",
                    "metric": "cuda_peak_reserved_max_mb",
                    "value": float(timing_df["max_cuda_peak_reserved_mb"].max()),
                },
            ]
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "operational_summary.csv", index=False)
    pd.DataFrame(memory_rows).to_csv(output_dir / "analysis_memory_usage.csv", index=False)

    per_object = timing_df.copy()
    denom = per_object["object_count"].clip(lower=1)
    for col in ["seg_sec", "classifi_sec", "normal_sec", "prior_sec", "run_sec", "run_plus_debug_sec"]:
        per_object[f"{col}_per_object"] = per_object[col] / denom
    per_object.to_csv(output_dir / "timing_per_object_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rate_metrics = summary_df[summary_df["group"].isin(["quality", "target"])].copy()
    rate_metrics = rate_metrics[rate_metrics["metric"].str.endswith("_rate")]
    axes[0].barh(rate_metrics["metric"], rate_metrics["value"], color="#72B7B2")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_title("Operational Rates")
    axes[0].set_xlabel("rate")

    axes[1].scatter(timing_df["object_count"], timing_df["run_plus_debug_sec"], alpha=0.7)
    axes[1].set_title("Runtime vs Object Count")
    axes[1].set_xlabel("object_count")
    axes[1].set_ylabel("run_plus_debug_sec")
    fig.tight_layout()
    fig.savefig(output_dir / "operational_rates_runtime.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    per_object_cols = ["seg_sec_per_object", "classifi_sec_per_object", "normal_sec_per_object", "prior_sec_per_object"]
    axes[0].boxplot([per_object[col].dropna().values for col in per_object_cols], tick_labels=per_object_cols, showfliers=False)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Stage Time per Object")
    axes[0].set_ylabel("seconds/object")

    memory_df = pd.DataFrame(memory_rows)
    axes[1].plot(memory_df["stage"], memory_df["rss_mb"], marker="o", label="rss_mb")
    axes[1].plot(memory_df["stage"], memory_df["peak_rss_mb"], marker="o", label="peak_rss_mb")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_title("Analysis Script Memory")
    axes[1].set_ylabel("MB")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_object_time_memory.png", dpi=160)
    plt.close(fig)


def main() -> int:
    start_time = time.perf_counter()
    memory_rows: list[dict[str, Any]] = [
        {"stage": "start", "rss_mb": current_rss_mb(), "peak_rss_mb": peak_rss_mb()},
    ]
    parser = argparse.ArgumentParser(description="Analyze debug_result JSON files and save report CSV/PNG files.")
    parser.add_argument("--input", default="debug_result", help="debug_result directory")
    parser.add_argument("--pattern", default="*_result.json", help="json glob pattern")
    parser.add_argument("--output", default="analysis_report", help="report output directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_dir, args.pattern)
    memory_rows.append({"stage": "after_load_json", "rss_mb": current_rss_mb(), "peak_rss_mb": peak_rss_mb()})
    if not records:
        raise RuntimeError(f"no result json files found: {input_dir}/{args.pattern}")

    timing_df = build_timing_df(records)
    inference_memory_df = build_inference_memory_df(records)
    obj_df = build_object_df(records)
    memory_rows.append({"stage": "after_build_tables", "rss_mb": current_rss_mb(), "peak_rss_mb": peak_rss_mb()})
    if obj_df.empty:
        raise RuntimeError("no objects found in result json files")

    timing_df.to_csv(output_dir / "timing_metrics.csv", index=False)
    obj_df.to_csv(output_dir / "object_metrics.csv", index=False)
    save_timing_reports(timing_df, output_dir)
    save_inference_memory_reports(inference_memory_df, timing_df, output_dir)
    save_class_reports(obj_df, output_dir)
    save_priority_reports(obj_df, output_dir)
    save_target_neighbor_surface_reports(obj_df, output_dir)
    memory_rows.append({"stage": "after_plots", "rss_mb": current_rss_mb(), "peak_rss_mb": peak_rss_mb()})
    save_operational_reports(timing_df, obj_df, output_dir, memory_rows, time.perf_counter() - start_time)

    print(f"loaded files: {len(records)}")
    print(f"objects: {len(obj_df)}")
    print(f"inference memory snapshots: {len(inference_memory_df)}")
    if not inference_memory_df.empty:
        for col in ["rss_mb", "peak_rss_mb", "cuda_allocated_mb", "cuda_reserved_mb", "cuda_max_allocated_mb", "cuda_max_reserved_mb"]:
            if col in inference_memory_df.columns and inference_memory_df[col].notna().any():
                print(
                    f"{col}: mean={inference_memory_df[col].mean():.1f} MB, "
                    f"p95={inference_memory_df[col].quantile(0.95):.1f} MB, "
                    f"max={inference_memory_df[col].max():.1f} MB"
                )
    print(f"analysis peak rss: {peak_rss_mb():.1f} MB")
    print(f"saved report files to: {output_dir}")
    for path in sorted(output_dir.glob("*")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
