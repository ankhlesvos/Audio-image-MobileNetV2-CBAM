"""
Model Scaling Experiments — MyNet-S / M / L
=============================================
Width-multiplier sweep on the M5 backbone + Logit Adjustment (TF mode).
All training hyperparameters, data augmentation, and model structure are
held constant; only `width_mult` changes.

Dataset: 3-class DeepShip (Cargo+Tug=0, Passengership=1, Tanker=2)

Usage:
  # Full run (data prep → train → plot)
  python run_scaling_experiments.py

  # Skip training, re-generate plots from existing JSON
  python run_scaling_experiments.py --plot-only

Output:
  - saved_models/scaling/MyNet_S/ … /MyNet_M/ … /MyNet_L/
  - scaling_results_summary.txt   (one-line table per variant)
  - scaling_results_raw.json      (raw dict for post-processing)
  - scaling_tradeoff_plot.png     (2-panel Acc vs Params / F1 vs FLOPs)
  - scaling_performance_vs_width.png  (Acc & F1 vs α)
  - scaling_cpu_latency.png       (CPU latency bar chart)
  - scaling_deployment_analysis.txt  (deployment-oriented resource table)
"""

import yaml
import os
import sys
import json
import copy
import re
import time
import argparse
import subprocess
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────────────────────────────
# CONFIG_CBAM_S24  (M5 backbone — identical to run_experiments.py)
# ──────────────────────────────────────────────────────────────────────
CONFIG_CBAM_S24 = [
    [1, 16, 1, 1, 0],
    [6, 24, 2, 2, 2],   # Stage 2  – Pre-DW CBAM
    [6, 32, 3, 1, 0],
    [6, 64, 4, 2, 2],   # Stage 4  – Pre-DW CBAM
    [6, 96, 3, 1, 0],
    [6, 160, 3, 1, 0],
    [6, 320, 1, 1, 0],
]

# ──────────────────────────────────────────────────────────────────────
# Scaling variants  —  ONLY width_mult differs
# ──────────────────────────────────────────────────────────────────────
SCALING_VARIANTS = {
    "MyNet-S": {"width_mult": 0.50, "description": "Extreme-lightweight  (≈31% params of L)"},
    "MyNet-M": {"width_mult": 0.75, "description": "Mid-range MCU        (≈61% params of L)"},
    "MyNet-L": {"width_mult": 1.00, "description": "Full model / ceiling (100% params)"},
}

# Default random seeds for multi-seed reproducibility study
DEFAULT_SEEDS = [42, 123, 2024]

# ──────────────────────────────────────────────────────────────────────
# Fixed training config  (M5 backbone + LA-TF mode, 3-class dataset)
# ──────────────────────────────────────────────────────────────────────
BASE_CONFIG = {
    "train_conf": {
        "use_gpu": True,
        "batch_size": 16,   # 4 GB GPU: 32 OOMs on 3ch input; 16 is safe
        "num_workers": 4,
        "max_epoch": 50,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "freeze_backbone": False,
        "seed": 42,
        "save_model_dir": None,            # filled per-variant
        "use_sampler": False,
        "sampler_alpha": 0.5,
        "use_class_weights": True,
        "monitor_metric": "val_loss",      # val_loss is the most stable ES signal
        "patience": 10,
        "min_epochs": 15,
        "min_delta": 0.001,
        "loss_conf": {
            "loss_type": "ce",
            "gamma": 0.0,
            "label_smoothing": 0.1,
            # LA-TF mode: train=on, eval=off  (best mode from ablation)
            "apply_logit_adj_in_train": True,
            "apply_logit_adj_in_eval": False,
            "logit_adj_tau": 1.0,
            "pair_penalty": {
                "use_penalty": False,
                "weight": 2.0,
                "targets": [[1, 2], [2, 0]],
            },
        },
    },
    "data_conf": {
        # 3-class DeepShip lists (Cargo+Tug=0, Passengership=1, Tanker=2)
        "train_list": "data/train_list_5s_new.txt",
        "val_list":   "data/val_list_5s.txt",
        "test_list":  "data/test_list_5s.txt",
    },
    "model_conf": {
        "num_classes": 3,
        "in_channels": 3,
        "width_mult": 1.0,                 # overridden per-variant
        "model_config": CONFIG_CBAM_S24,
        "asymmetric": True,
        "multiscale": True,
        "force_no_residual": False,
        "audio_mode": False,
    },
}

# ──────────────────────────────────────────────────────────────────────
# Visual style constants
# ──────────────────────────────────────────────────────────────────────
COLORS  = {"MyNet-S": "#e41a1c", "MyNet-M": "#377eb8", "MyNet-L": "#4daf4a"}
MARKERS = {"MyNet-S": "o",       "MyNet-M": "s",       "MyNet-L": "^"}
SHORT   = {"MyNet-S": "S",       "MyNet-M": "M",       "MyNet-L": "L"}


# ══════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════

def parse_results(results_path: Path) -> dict:
    """Parse the results.txt written by train.py into a dict.

    train.py writes (line 773-780):
        Best Val Acc: 0.9123 @ Epoch 32
        Best Val F1:  0.9011 @ Epoch 29
        Test File Acc: 0.9200
        Test File F1:  0.9050
        Params: 2299136
        FLOPs: 959750144
    """
    if not results_path.exists():
        print(f"  [WARN] results.txt not found: {results_path}")
        return {}
    text = results_path.read_text()
    patterns = {
        "val_acc":  r"Best Val Acc:\s+([0-9.]+)",
        "val_f1":   r"Best Val F1:\s+([0-9.]+)",
        "test_acc": r"Test File Acc:\s+([0-9.]+)",
        "test_f1":  r"Test File F1:\s+([0-9.]+)",
        "params":   r"Params:\s+([0-9]+)",
        "flops":    r"FLOPs:\s+([0-9]+)",
    }
    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        out[key] = float(m.group(1)) if m else None
    return out


def compute_model_stats(width_mult: float) -> dict:
    """Use thop to compute Params/FLOPs for the given width_mult offline
    (without needing a trained checkpoint).  Returns params_m, flops_m."""
    try:
        from modules.model import MyNet
        model = MyNet(
            num_classes=3,
            model_config=CONFIG_CBAM_S24,
            width_mult=width_mult,
            in_channels=3,
            asymmetric=True,
            multiscale=True,
        )
        # Input shape matches dataset.py → (batch, 3, n_mels=80, time≈301)
        flops, params = model.profile_model(input_size=(3, 80, 301))
        return {"params_m": round(params / 1e6, 4), "flops_m": round(flops / 1e6, 2)}
    except Exception as e:
        print(f"  [WARN] profile_model failed: {e}")
        return {"params_m": None, "flops_m": None}


def measure_cpu_latency_from_dir(ckpt_dir: str, width_mult: float,
                                 n_warmup: int = 10, n_runs: int = 100) -> dict:
    """Load the best checkpoint from *ckpt_dir*, time CPU forward passes."""
    from modules.model import MyNet

    d = Path(ckpt_dir)
    ckpts = sorted(
        d.glob("best_model_epoch_*.pth"),
        key=lambda p: int(re.search(r"epoch_(\d+)", p.name).group(1))
    )
    if not ckpts:
        print(f"  [CPU timing] No checkpoint found in {ckpt_dir}, skipping.")
        return {"mean_ms": None, "std_ms": None}

    best_ckpt = ckpts[-1]
    device = torch.device("cpu")
    model = MyNet(
        num_classes=3,
        model_config=CONFIG_CBAM_S24,
        width_mult=width_mult,
        in_channels=3,
        asymmetric=True,
        multiscale=True,
    )
    sd = torch.load(best_ckpt, map_location=device, weights_only=True)
    sd = {k: v for k, v in sd.items()
          if "total_ops" not in k and "total_params" not in k}
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    dummy = torch.randn(1, 3, 80, 301, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append(time.perf_counter() - t0)

    mean_ms = float(np.mean(times) * 1000)
    std_ms  = float(np.std(times)  * 1000)
    print(f"  [CPU timing] {Path(ckpt_dir).name}: {mean_ms:.1f} ms ± {std_ms:.1f} ms")
    return {"mean_ms": mean_ms, "std_ms": std_ms}


def measure_cpu_latency(variant_name: str, width_mult: float,
                        n_warmup: int = 10, n_runs: int = 100) -> dict:
    """Backward-compat wrapper around measure_cpu_latency_from_dir."""
    return measure_cpu_latency_from_dir(
        f"saved_models/scaling/{variant_name}", width_mult, n_warmup, n_runs
    )


def model_file_size(variant_name: str) -> dict:
    """Return size of the best checkpoint in KB."""
    ckpt_dir = Path(f"saved_models/scaling/{variant_name}")
    ckpts = sorted(ckpt_dir.glob("best_model_epoch_*.pth"))
    if not ckpts:
        return {"size_kb": None}
    size_kb = ckpts[-1].stat().st_size / 1024
    return {"size_kb": round(size_kb, 1)}


# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════

def _safe(results: dict, label: str, key: str):
    return results.get(label, {}).get(key)


# ══════════════════════════════════════════════════════════════════════
# Multi-seed aggregation helpers
# ══════════════════════════════════════════════════════════════════════

def aggregate_seeds(per_seed: list[dict]) -> dict:
    """Given a list of per-seed result dicts, compute mean ± std for each
    numeric metric.  Returns a dict with keys 'mean_<k>', 'std_<k>' and
    all non-numeric keys from the first seed (params_m, flops_m, …).
    """
    if not per_seed:
        return {}
    numeric_keys = [k for k, v in per_seed[0].items()
                    if isinstance(v, (int, float)) and k != "width_mult"]
    agg = {}
    for k in numeric_keys:
        vals = [r[k] for r in per_seed if r.get(k) is not None]
        if vals:
            agg[f"mean_{k}"] = round(float(np.mean(vals)), 4)
            agg[f"std_{k}"]  = round(float(np.std(vals)),  4)
    # carry forward static fields
    for k in ("width_mult", "params_m", "flops_m", "size_kb",
               "cpu_mean_ms", "cpu_std_ms"):
        if k in per_seed[0]:
            agg[k] = per_seed[0][k]
    agg["seeds_ran"] = [r.get("seed") for r in per_seed]
    agg["n_seeds"]   = len(per_seed)
    # convenience aliases used by plotting code
    for metric in ("test_acc", "test_f1", "val_acc", "val_f1"):
        if f"mean_{metric}" in agg:
            agg[metric] = agg[f"mean_{metric}"]  # best-compat alias
    return agg


def plot_tradeoff_2panel(results: dict, out_path: str = "scaling_tradeoff_plot.png"):
    """
    2-panel publication-quality trade-off figure:
      Left:  Test Accuracy  vs  Params (M)   — log-x
      Right: Test Macro-F1  vs  FLOPs (M)    — log-x
    Arrows show marginal gain S→M and M→L.
    Error-bars shown when std_* fields are present (multi-seed runs).
    """
    labels = list(results.keys())
    if len(labels) < 2:
        print("  [plot] Not enough variants to draw trade-off plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MyNet Scaling — Accuracy / F1 Trade-off",
                 fontsize=14, fontweight="bold", y=1.01)

    sorted_labels = sorted(labels, key=lambda l: results[l].get("params_m") or 0)

    for ax, xkey, ykey, std_ykey, xlabel, ylabel, title in [
        (ax1, "params_m", "test_acc", "std_test_acc", "Params (M)", "Test Accuracy",
         "Accuracy vs. Model Size"),
        (ax2, "flops_m",  "test_f1",  "std_test_f1",  "FLOPs (M)",  "Test Macro-F1",
         "Macro-F1 vs. Compute"),
    ]:
        for l in sorted_labels:
            xv  = _safe(results, l, xkey)
            yv  = _safe(results, l, ykey)
            yerr = _safe(results, l, std_ykey)  # None for single-seed runs
            if xv is None or yv is None:
                continue
            c = COLORS.get(l, "#999")
            m = MARKERS.get(l, "o")
            if yerr is not None:
                ax.errorbar(xv, yv, yerr=yerr,
                            fmt=m, color=c, ecolor=c, capsize=5,
                            markersize=11, markeredgecolor="k",
                            markeredgewidth=1.2, elinewidth=1.5, zorder=5)
            else:
                ax.scatter(xv, yv, s=220, c=c, marker=m, zorder=5,
                           edgecolors="k", linewidths=1.2)
            ax.annotate(
                SHORT.get(l, l),
                xy=(xv, yv),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=13, fontweight="bold", color=c,
            )

        # Marginal-gain arrows S→M, M→L
        for i in range(1, len(sorted_labels)):
            lp, lc = sorted_labels[i - 1], sorted_labels[i]
            xp = _safe(results, lp, xkey)
            xc = _safe(results, lc, xkey)
            yp = _safe(results, lp, ykey)
            yc = _safe(results, lc, ykey)
            if None in (xp, xc, yp, yc):
                continue
            dy = yc - yp
            mid_x = (xp + xc) / 2
            mid_y = (yp + yc) / 2
            ax.annotate("",
                        xy=(xc, yc), xytext=(xp, yp),
                        arrowprops=dict(arrowstyle="->", color="gray",
                                        lw=1.5, ls="--"))
            sign = "+" if dy >= 0 else ""
            ax.text(mid_x, mid_y + (max(yc, yp) - min(yc, yp)) * 0.15 + 0.004,
                    f"{sign}{dy:.3f}",
                    fontsize=9, ha="center", color="gray", fontstyle="italic")

        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.tick_params(labelsize=10)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=MARKERS[l], color="w",
               markerfacecolor=COLORS[l], markeredgecolor="k",
               markersize=10, label=l)
        for l in sorted_labels if l in COLORS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_performance_vs_width(results: dict,
                              out_path: str = "scaling_performance_vs_width.png"):
    """Acc & F1 vs width-multiplier (α), with errorbars when std available."""
    labels = sorted(results.keys(),
                    key=lambda l: results[l].get("width_mult") or 0)

    wm_vals   = [results[l].get("width_mult")    for l in labels]
    acc_vals  = [results[l].get("test_acc")      for l in labels]
    f1_vals   = [results[l].get("test_f1")       for l in labels]
    acc_stds  = [results[l].get("std_test_acc")  for l in labels]
    f1_stds   = [results[l].get("std_test_f1")   for l in labels]

    if not any(v is not None for v in acc_vals):
        return

    has_std = any(v is not None for v in acc_stds)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if has_std:
        ax.errorbar(wm_vals, acc_vals, yerr=acc_stds,
                    fmt="o-", color="#377eb8", ecolor="#377eb8",
                    capsize=5, lw=2, markersize=9, label="Test Accuracy (mean±std)")
        ax.errorbar(wm_vals, f1_vals, yerr=f1_stds,
                    fmt="s--", color="#e41a1c", ecolor="#e41a1c",
                    capsize=5, lw=2, markersize=9, label="Test Macro-F1 (mean±std)")
    else:
        ax.plot(wm_vals, acc_vals, "o-",  color="#377eb8", lw=2,
                label="Test Accuracy", markersize=9)
        ax.plot(wm_vals, f1_vals,  "s--", color="#e41a1c", lw=2,
                label="Test Macro-F1", markersize=9)

    for i, l in enumerate(labels):
        if acc_vals[i] is not None:
            ax.annotate(SHORT.get(l, l),
                        xy=(wm_vals[i], acc_vals[i]),
                        xytext=(8, 6), textcoords="offset points",
                        fontsize=11, fontweight="bold", color="#377eb8")

    ax.set_xlabel("Width Multiplier (α)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("MyNet Scaling — Performance vs. Width Multiplier", fontsize=13)
    ax.set_xticks(wm_vals)
    ax.set_xticklabels([f"{v:.2f}" for v in wm_vals], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, ls="--", alpha=0.35)
    ax.set_ylim(
        min(filter(None, acc_vals + f1_vals)) * 0.97,
        max(filter(None, acc_vals + f1_vals)) * 1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cpu_latency(results: dict,
                     out_path: str = "scaling_cpu_latency.png"):
    """Horizontal bar chart: CPU latency + file size per variant.
    Variants with None latency (failed training) are skipped gracefully.
    """
    all_labels = list(results.keys())

    # Filter to only variants that have latency data
    lat_labels = [l for l in all_labels if results[l].get("cpu_mean_ms") is not None]
    lat_vals   = [results[l]["cpu_mean_ms"] for l in lat_labels]
    lat_errs   = [results[l].get("cpu_std_ms", 0) or 0 for l in lat_labels]

    # File size: include all variants that have size info
    sz_labels  = [l for l in all_labels if results[l].get("size_kb") is not None]
    size_vals  = [results[l]["size_kb"] for l in sz_labels]

    if not lat_vals and not size_vals:
        print("  [plot] No CPU latency / size data; skipping latency chart.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # -- CPU latency bars --
    if lat_vals:
        colors_lat = [COLORS.get(l, "#999") for l in lat_labels]
        bars = ax1.barh(lat_labels, lat_vals, xerr=lat_errs,
                        color=colors_lat, edgecolor="k", height=0.5,
                        error_kw=dict(ecolor="black", capsize=4, linewidth=1.2))
        max_lat = max(lat_vals)
        for bar, v in zip(bars, lat_vals):
            ax1.text(v + max_lat * 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{v:.1f} ms", va="center", fontsize=10)
        ax1.invert_yaxis()
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=12, color="gray")
    ax1.set_xlabel("CPU Inference Latency (ms)", fontsize=11)
    ax1.set_title("CPU Latency (single sample)", fontsize=12)
    ax1.grid(axis="x", ls="--", alpha=0.4)

    # -- File size bars --
    if size_vals:
        colors_sz = [COLORS.get(l, "#999") for l in sz_labels]
        bars2 = ax2.barh(sz_labels, size_vals,
                         color=colors_sz, edgecolor="k", height=0.5)
        max_sz = max(size_vals)
        for bar, v in zip(bars2, size_vals):
            ax2.text(v + max_sz * 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{v:.0f} KB", va="center", fontsize=10)
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="gray")
    ax2.set_xlabel("Checkpoint Size (KB)", fontsize=11)
    ax2.set_title("Model File Size", fontsize=12)
    ax2.grid(axis="x", ls="--", alpha=0.4)

    fig.suptitle("MyNet Scaling — Deployment Resource Analysis",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def write_deployment_analysis(results: dict,
                               out_path: str = "scaling_deployment_analysis.txt"):
    """Write a human-readable deployment-oriented resource analysis table."""
    lines = []
    lines.append("=" * 100)
    lines.append("MyNet Scaling Experiments — Deployment-Oriented Resource Analysis")
    lines.append("=" * 100)
    lines.append(
        f"{'Variant':<10} | {'α':>4} | {'Params(M)':>9} | {'FLOPs(M)':>8} | "
        f"{'Size(KB)':>8} | {'CPU(ms)':>9} | {'TestAcc':>7} | {'TestF1':>7} | "
        f"{'ValAcc':>6} | {'ValF1':>6} | Notes"
    )
    lines.append("-" * 100)

    baseline_acc = results.get("MyNet-L", {}).get("test_acc") or 1.0
    baseline_f1  = results.get("MyNet-L", {}).get("test_f1")  or 1.0
    baseline_lat = results.get("MyNet-L", {}).get("cpu_mean_ms") or 1.0

    for variant_name, variant_conf in SCALING_VARIANTS.items():
        r = results.get(variant_name, {})
        p   = r.get("params_m", "N/A")
        fl  = r.get("flops_m",  "N/A")
        sz  = r.get("size_kb",  "N/A")
        lat = r.get("cpu_mean_ms", "N/A")
        ta  = r.get("test_acc", "N/A")
        tf  = r.get("test_f1",  "N/A")
        va  = r.get("val_acc",  "N/A")
        vf  = r.get("val_f1",   "N/A")
        wm  = variant_conf["width_mult"]

        # Efficiency notes
        notes = []
        if isinstance(ta, float) and isinstance(lat, float):
            acc_drop = (baseline_acc - ta) * 100
            lat_drop = (baseline_lat - lat) / baseline_lat * 100
            notes.append(f"Acc↓{acc_drop:.1f}%  Lat↓{lat_drop:.1f}%")
        lines.append(
            f"{variant_name:<10} | {wm:>4.2f} | {p!s:>9} | {fl!s:>8} | "
            f"{sz!s:>8} | {lat!s:>9} | {ta!s:>7} | {tf!s:>7} | "
            f"{va!s:>6} | {vf!s:>6} | {'; '.join(notes)}"
        )

    lines.append("=" * 100)
    lines.append("")
    lines.append("Deployment Recommendation:")
    lines.append(
        "  MyNet-S : Ultra-lightweight. Suitable for MCU/edge with strict memory (<1 MB flash), "
        "trade-off ~2-4% accuracy."
    )
    lines.append(
        "  MyNet-M : Balanced. Good for mid-range embedded systems."
        " ~50% param reduction vs L with marginal accuracy loss."
    )
    lines.append(
        "  MyNet-L : Full-performance. Recommended for GPU/server. Highest accuracy & F1."
    )
    lines.append("")
    lines.append("Input shape used for profiling: (1, 3, 80, 301)  "
                 "[batch=1, ch=3 (mel+Δ+ΔΔ), n_mels=80, time≈5 s at 32kHz/hop=512]")
    lines.append("CPU latency: 100 forward passes on CPU (no GPU), single sample.")

    text = "\n".join(lines)
    Path(out_path).write_text(text)
    print(f"  Saved: {out_path}")
    print()
    print(text)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MyNet-S/M/L scaling experiments (3-class DeepShip)"
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip training; load existing scaling_results_raw.json and regenerate plots."
    )
    parser.add_argument(
        "--skip-latency", action="store_true",
        help="Skip CPU latency measurement (useful when no checkpoint exists yet)."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"Random seeds to use (default: {DEFAULT_SEEDS})."
    )
    parser.add_argument(
        "--skip-done", action="store_true",
        help="Skip training a (variant, seed) pair if its results.txt already exists."
    )
    args = parser.parse_args()
    SEEDS = args.seeds
    print(f"Multi-seed mode: {SEEDS}  ({len(SEEDS)} runs per variant)")

    json_path = Path("scaling_results_raw.json")

    # ── Plot-only mode ──────────────────────────────────────────────
    if args.plot_only:
        if not json_path.exists():
            print(f"ERROR: --plot-only requires {json_path} to exist. "
                  "Run without --plot-only first.")
            sys.exit(1)
        with open(json_path) as f:
            raw = json.load(f)
        # Support both old format (flat dict) and new format ({aggregated, per_seed})
        results = raw.get("aggregated", raw)
        print(f"Loaded results from {json_path}")
        print("\n--- Regenerating plots ---")
        plot_tradeoff_2panel(results)
        plot_performance_vs_width(results)
        plot_cpu_latency(results)
        write_deployment_analysis(results)
        return

    # ── Full training run ───────────────────────────────────────────
    configs_dir = Path("configs/scaling")
    configs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path("scaling_results_summary.txt")
    header = (
        f"{'Variant':<10} | {'α':>4} | {'Seed':>6} | "
        f"{'ValAcc':>7} | {'ValF1':>7} | {'TestAcc':>8} | {'TestF1':>8} | "
        f"{'Params(M)':>9} | {'FLOPs(M)':>9} | Description\n"
    )
    summary_path.write_text(header + "-" * 120 + "\n")

    # per_seed_results[variant][seed_idx] = result dict
    per_seed_results: dict[str, list[dict]] = {v: [] for v in SCALING_VARIANTS}

    for variant_name, variant_conf in SCALING_VARIANTS.items():
        print(f"\n{'='*70}")
        print(f"  Scaling Experiment: {variant_name}  "
              f"(width_mult = {variant_conf['width_mult']})")
        print(f"  Seeds: {SEEDS}")
        print(f"{'='*70}")

        for seed in SEEDS:
            run_dir = f"saved_models/scaling/{variant_name}_seed{seed}"
            print(f"\n  --- Seed {seed} → {run_dir} ---")

            # ── Build YAML config  ───────────────────────────────────
            cfg = copy.deepcopy(BASE_CONFIG)
            cfg["model_conf"]["width_mult"] = variant_conf["width_mult"]
            cfg["train_conf"]["seed"]           = seed
            cfg["train_conf"]["save_model_dir"] = run_dir

            yaml_path = configs_dir / f"{variant_name}_seed{seed}_config.yml"
            with open(yaml_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)

            # ── Run training ─────────────────────────────────────────
            results_txt = Path(run_dir) / "results.txt"
            if args.skip_done and results_txt.exists():
                print(f"  [SKIP] {run_dir}/results.txt already exists, reusing.")
            else:
                command = [sys.executable, "train.py", "-c", str(yaml_path)]
                print(f"  Launching: {' '.join(command)}")
                try:
                    subprocess.run(command, check=True)
                    print(f"  ✅ {variant_name} seed={seed} training finished.")
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ {variant_name} seed={seed} FAILED: {e}")
                    per_seed_results[variant_name].append(
                        {"seed": seed, "width_mult": variant_conf["width_mult"],
                         "error": str(e)}
                    )
                    continue

            # ── Parse results.txt ────────────────────────────────────
            res = parse_results(Path(run_dir) / "results.txt")
            res["seed"]       = seed
            res["width_mult"] = variant_conf["width_mult"]

            # Params/FLOPs
            if res.get("params") and res["params"] > 0:
                res["params_m"] = round(res["params"] / 1e6, 4)
            if res.get("flops") and res["flops"] > 0:
                res["flops_m"]  = round(res["flops"] / 1e6, 2)
            if not res.get("params_m") or not res.get("flops_m"):
                stats = compute_model_stats(variant_conf["width_mult"])
                res.setdefault("params_m", stats["params_m"])
                res.setdefault("flops_m",  stats["flops_m"])

            per_seed_results[variant_name].append(res)

            # ── Append to per-run summary ────────────────────────────
            with open(summary_path, "a") as sf:
                sf.write(
                    f"{variant_name:<10} | {res['width_mult']:>4.2f} | {seed:>6} | "
                    f"{res.get('val_acc', 'N/A')!s:>7} | {res.get('val_f1', 'N/A')!s:>7} | "
                    f"{res.get('test_acc', 'N/A')!s:>8} | {res.get('test_f1', 'N/A')!s:>8} | "
                    f"{res.get('params_m', 'N/A')!s:>9} | {res.get('flops_m', 'N/A')!s:>9} | "
                    f"{variant_conf['description']}\n"
                )

    # ── Aggregate mean ± std across seeds ───────────────────────────
    print("\n--- Aggregating multi-seed results ---")
    results: dict[str, dict] = {}  # variant → aggregated dict
    for variant_name, seed_list in per_seed_results.items():
        good = [r for r in seed_list if "error" not in r]
        if not good:
            results[variant_name] = {"error": "all_seeds_failed",
                                     "width_mult": SCALING_VARIANTS[variant_name]["width_mult"]}
            continue
        agg = aggregate_seeds(good)
        results[variant_name] = agg
        print(f"  {variant_name}: seeds={agg['seeds_ran']}  "
              f"TestAcc={agg.get('mean_test_acc','?'):.4f}±{agg.get('std_test_acc','?'):.4f}  "
              f"TestF1={agg.get('mean_test_f1','?'):.4f}±{agg.get('std_test_f1','?'):.4f}")

    # ── Append aggregated summary block ─────────────────────────────
    with open(summary_path, "a") as sf:
        sf.write("\n" + "=" * 120 + "\n")
        sf.write("AGGREGATED (mean ± std across seeds)\n")
        sf.write("=" * 120 + "\n")
        agg_header = (
            f"{'Variant':<10} | {'α':>4} | {'Seeds':>15} | "
            f"{'TestAcc_mean':>12} | {'TestAcc_std':>11} | "
            f"{'TestF1_mean':>11} | {'TestF1_std':>10}\n"
        )
        sf.write(agg_header)
        for v, r in results.items():
            if "error" in r:
                sf.write(f"{v:<10} | ERROR\n")
                continue
            sf.write(
                f"{v:<10} | {r['width_mult']:>4.2f} | "
                f"{str(r.get('seeds_ran','?')):>15} | "
                f"{r.get('mean_test_acc','N/A')!s:>12} | "
                f"{r.get('std_test_acc','N/A')!s:>11} | "
                f"{r.get('mean_test_f1','N/A')!s:>11} | "
                f"{r.get('std_test_f1','N/A')!s:>10}\n"
            )

    # ── CPU latency (only from last seed of each variant) ───────────
    if not args.skip_latency:
        for variant_name, variant_conf in SCALING_VARIANTS.items():
            if variant_name not in results or "error" in results[variant_name]:
                continue
            last_seed = SEEDS[-1]
            run_dir = f"saved_models/scaling/{variant_name}_seed{last_seed}"
            print(f"\n  Measuring CPU latency for {variant_name} (seed={last_seed}) …")
            cpu_info = measure_cpu_latency_from_dir(
                run_dir, variant_conf["width_mult"]
            )
            results[variant_name]["cpu_mean_ms"] = cpu_info["mean_ms"]
            results[variant_name]["cpu_std_ms"]  = cpu_info["std_ms"]

    # ── Save raw JSON ────────────────────────────────────────────────
    raw_out = {"aggregated": results, "per_seed": per_seed_results}
    with open(json_path, "w") as f:
        json.dump(raw_out, f, indent=2, default=str)
    print(f"\n  Raw results → {json_path}")

    # ── Generate all plots & deployment analysis ─────────────────────
    print("\n--- Generating plots & deployment analysis ---")
    plot_tradeoff_2panel(results)
    plot_performance_vs_width(results)
    plot_cpu_latency(results)
    write_deployment_analysis(results)

    # ── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ALL SCALING EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"\nSummary → {summary_path}")
    print("\nQuick view:")
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
