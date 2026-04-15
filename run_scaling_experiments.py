"""
Model Scaling Experiments — MyNet-S / M / L
=============================================
Width-multiplier sweep on the M5 backbone + Logit Adjustment (TF mode).
All training hyperparameters, data augmentation, and model structure are
held constant; only `width_mult` changes.

Output:
  - saved_models/scaling/MyNet_S/ … /MyNet_M/ … /MyNet_L/
  - scaling_results_summary.txt   (one-line table per variant)
  - scaling_tradeoff_plot.png     (Accuracy vs. Params & F1 vs. FLOPs)
  - scaling_cpu_latency.json      (CPU inference timing per variant)
"""

import yaml
import os
import sys
import json
import copy
import re
import time
import subprocess
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
# CONFIG_CBAM_S24  (identical to run_experiments.py)
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
    "MyNet-S": {"width_mult": 0.50, "description": "Extreme-resource MCU  (≈25% params of L)"},
    "MyNet-M": {"width_mult": 0.75, "description": "Mid-range MCU        (≈56% params of L)"},
    "MyNet-L": {"width_mult": 1.00, "description": "Full model / ceiling (100% params)"},
}

# ──────────────────────────────────────────────────────────────────────
# Fixed training config  (M5 backbone + LA-TF mode)
# ──────────────────────────────────────────────────────────────────────
BASE_CONFIG = {
    "train_conf": {
        "use_gpu": True,
        "batch_size": 32,
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
        "monitor_metric": "f1",
        "patience": 10,
        "loss_conf": {
            "loss_type": "ce",
            "gamma": 0.0,
            "label_smoothing": 0.1,
            # LA-TF mode: train=on, eval=off
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
# Helpers
# ──────────────────────────────────────────────────────────────────────
def parse_results(results_path: Path) -> dict:
    """Parse the results.txt written by train.py into a dict."""
    if not results_path.exists():
        return {}
    text = results_path.read_text()
    out = {}
    patterns = {
        "val_acc":  r"Best Val Acc:\s+([0-9.]+)",
        "val_f1":   r"Best Val F1:\s+([0-9.]+)",
        "test_acc": r"Test File Acc:\s+([0-9.]+)",
        "test_f1":  r"Test File F1:\s+([0-9.]+)",
        "params":   r"Params:\s+([0-9.]+)",
        "flops":    r"FLOPs:\s+([0-9.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        out[key] = float(m.group(1)) if m else None
    return out


def measure_cpu_latency(variant_name: str, width_mult: float) -> dict:
    """
    Load the best checkpoint for *variant_name*, move to CPU, and time
    100 forward passes with torch.no_grad().  Returns {mean_ms, std_ms}.
    """
    from modules.model import MyNet

    ckpt_dir = Path(f"saved_models/scaling/{variant_name}")
    # Find the best model (highest-val F1 checkpoint)
    ckpts = sorted(ckpt_dir.glob("best_model_epoch_*.pth"))
    if not ckpts:
        print(f"  [CPU timing] No checkpoint found for {variant_name}, skipping.")
        return {"mean_ms": None, "std_ms": None}

    best_ckpt = ckpts[-1]  # last saved = best (train.py saves in order)
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
    sd = {k: v for k, v in sd.items() if "total_ops" not in k and "total_params" not in k}
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    dummy = torch.randn(1, 3, 160, 157, device=device)

    # warm-up
    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    # timing
    times = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            model(dummy)
            times.append(time.perf_counter() - t0)

    mean_ms = float(np.mean(times) * 1000)
    std_ms  = float(np.std(times)  * 1000)
    print(f"  [CPU timing] {variant_name}: {mean_ms:.1f} ms ± {std_ms:.1f}")
    return {"mean_ms": mean_ms, "std_ms": std_ms}


def model_file_size(variant_name: str) -> dict:
    """Return size of the best checkpoint in KB."""
    ckpt_dir = Path(f"saved_models/scaling/{variant_name}")
    ckpts = sorted(ckpt_dir.glob("best_model_epoch_*.pth"))
    if not ckpts:
        return {"size_kb": None}
    size_kb = ckpts[-1].stat().st_size / 1024
    return {"size_kb": round(size_kb, 1)}


def plot_tradeoffs(results: dict):
    """
    Produce a 1×2 figure:
      (left)  Test Acc  vs  Params (log-x)
      (right) Test F1   vs  FLOPs  (log-x)
    """
    labels = list(results.keys())
    if not labels:
        return

    # --- Figure 1: Accuracy vs Params ---
    params_vals = [results[l]["params_m"] for l in labels if results[l].get("params_m")]
    acc_vals    = [results[l]["test_acc"]  for l in labels if results[l].get("params_m")]
    param_labels = [l for l in labels if results[l].get("params_m")]

    if params_vals and acc_vals:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        for i, l in enumerate(param_labels):
            ax.scatter(params_vals[i], acc_vals[i], s=180, c=[colors[i]], zorder=5, edgecolors="k")
            ax.annotate(l, (params_vals[i], acc_vals[i]), textcoords="offset points",
                        xytext=(8, 8), fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlabel("Params (M)")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Accuracy vs. Model Size (Params)")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig("scaling_accuracy_vs_params.png", dpi=200)
        plt.close(fig)
        print("  Saved: scaling_accuracy_vs_params.png")

    # --- Figure 2: F1 vs FLOPs ---
    flops_vals = [results[l]["flops_m"] for l in labels if results[l].get("flops_m")]
    f1_vals    = [results[l]["test_f1"] for l in labels if results[l].get("flops_m")]
    flops_labels = [l for l in labels if results[l].get("flops_m")]

    if flops_vals and f1_vals:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for i, l in enumerate(flops_labels):
            ax.scatter(flops_vals[i], f1_vals[i], s=180, c=[colors[i]], zorder=5, edgecolors="k")
            ax.annotate(l, (flops_vals[i], f1_vals[i]), textcoords="offset points",
                        xytext=(8, 8), fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlabel("FLOPs (M)")
        ax.set_ylabel("Test Macro F1")
        ax.set_title("Macro F1 vs. FLOPs")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig("scaling_f1_vs_flops.png", dpi=200)
        plt.close(fig)
        print("  Saved: scaling_f1_vs_flops.png")

    # --- Figure 3: Combined trade-off (Accuracy & F1 vs. width_mult) ---
    wm_vals = [results[l]["width_mult"] for l in labels if results[l].get("width_mult")]
    acc_w   = [results[l]["test_acc"]   for l in labels if results[l].get("width_mult")]
    f1_w    = [results[l]["test_f1"]    for l in labels if results[l].get("width_mult")]
    lbl_w   = [l for l in labels if results[l].get("width_mult")]

    if wm_vals:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(wm_vals, acc_w, "o-", label="Test Accuracy", lw=2)
        ax.plot(wm_vals, f1_w,  "s--", label="Test Macro F1", lw=2)
        for i, l in enumerate(lbl_w):
            ax.annotate(l, (wm_vals[i], acc_w[i]), textcoords="offset points",
                        xytext=(8, 5), fontsize=10, fontweight="bold")
        ax.set_xlabel("Width Multiplier (α)")
        ax.set_ylabel("Score")
        ax.set_title("Performance vs. Width Multiplier")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_xticks(wm_vals)
        ax.set_xticklabels([f"{v:.2f}" for v in wm_vals])
        fig.tight_layout()
        fig.savefig("scaling_performance_vs_width.png", dpi=200)
        plt.close(fig)
        print("  Saved: scaling_performance_vs_width.png")


def export_tradeoff_plot(results: dict):
    """
    Single combined figure:
      Left panel:  Test Accuracy  vs  Params (log-x)
      Right panel: Test Macro F1  vs  FLOPs  (log-x)
    Points labelled S / M / L with arrows showing marginal gain.
    """
    labels = list(results.keys())
    if len(labels) < 2:
        return

    colors = {"MyNet-S": "#e41a1c", "MyNet-M": "#377eb8", "MyNet-L": "#4daf4a"}
    markers = {"MyNet-S": "o", "MyNet-M": "s", "MyNet-L": "^"}
    short = {"MyNet-S": "S", "MyNet-M": "M", "MyNet-L": "L"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Left: Accuracy vs Params ---
    for l in labels:
        r = results[l]
        if r.get("params_m") is None or r.get("test_acc") is None:
            continue
        c = colors.get(l, "#999999")
        m = markers.get(l, "o")
        ax1.scatter(r["params_m"], r["test_acc"], s=200, c=c, marker=m, zorder=5, edgecolors="k", linewidths=1)
        ax1.annotate(short.get(l, l), (r["params_m"], r["test_acc"]),
                     textcoords="offset points", xytext=(10, 10),
                     fontsize=13, fontweight="bold", color=c)

    # Marginal arrows
    sorted_labels = sorted(labels, key=lambda l: results[l].get("params_m", 0))
    for i in range(1, len(sorted_labels)):
        l_prev, l_cur = sorted_labels[i - 1], sorted_labels[i]
        r_p, r_c = results[l_prev], results[l_cur]
        if r_p.get("params_m") and r_c.get("params_m") and r_c.get("test_acc") and r_p.get("test_acc"):
            dx = r_c["params_m"] - r_p["params_m"]
            dy = r_c["test_acc"] - r_p["test_acc"]
            mid_x = r_p["params_m"] + dx / 2
            mid_y = r_p["test_acc"] + dy / 2
            ax1.annotate("", (r_c["params_m"], r_c["test_acc"]),
                         (r_p["params_m"], r_p["test_acc"]),
                         arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"))
            ax1.text(mid_x, mid_y + 0.005, f"+{dy:.3f}", fontsize=9,
                     ha="center", color="gray", fontstyle="italic")

    ax1.set_xscale("log")
    ax1.set_xlabel("Params (M)", fontsize=12)
    ax1.set_ylabel("Test Accuracy", fontsize=12)
    ax1.set_title("Accuracy vs. Params", fontsize=13)
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    # --- Right: F1 vs FLOPs ---
    for l in labels:
        r = results[l]
        if r.get("flops_m") is None or r.get("test_f1") is None:
            continue
        c = colors.get(l, "#999999")
        m = markers.get(l, "o")
        ax2.scatter(r["flops_m"], r["test_f1"], s=200, c=c, marker=m, zorder=5, edgecolors="k", linewidths=1)
        ax2.annotate(short.get(l, l), (r["flops_m"], r["test_f1"]),
                     textcoords="offset points", xytext=(10, 10),
                     fontsize=13, fontweight="bold", color=c)

    for i in range(1, len(sorted_labels)):
        l_prev, l_cur = sorted_labels[i - 1], sorted_labels[i]
        r_p, r_c = results[l_prev], results[l_cur]
        if r_p.get("flops_m") and r_c.get("flops_m") and r_c.get("test_f1") and r_p.get("test_f1"):
            dx = r_c["flops_m"] - r_p["flops_m"]
            dy = r_c["test_f1"] - r_p["test_f1"]
            mid_x = r_p["flops_m"] + dx / 2
            mid_y = r_p["test_f1"] + dy / 2
            ax2.annotate("", (r_c["flops_m"], r_c["test_f1"]),
                         (r_p["flops_m"], r_p["test_f1"]),
                         arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"))
            ax2.text(mid_x, mid_y + 0.005, f"+{dy:.3f}", fontsize=9,
                     ha="center", color="gray", fontstyle="italic")

    ax2.set_xscale("log")
    ax2.set_xlabel("FLOPs (M)", fontsize=12)
    ax2.set_ylabel("Test Macro F1", fontsize=12)
    ax2.set_title("Macro F1 vs. FLOPs", fontsize=13)
    ax2.grid(True, which="both", ls="--", alpha=0.4)

    fig.suptitle("MyNet Scaling — Accuracy / F1 Trade-off", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("scaling_tradeoff_plot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: scaling_tradeoff_plot.png")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    configs_dir = Path("configs/scaling")
    configs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path("scaling_results_summary.txt")
    with open(summary_path, "w") as sf:
        sf.write(
            f"{'Variant':<10} | {'α':>4} | {'Val Acc':>7} | {'Val F1':>7} | "
            f"{'Test Acc':>8} | {'Test F1':>8} | {'Params(M)':>9} | "
            f"{'FLOPs(M)':>8} | {'Size(KB)':>8} | {'CPU(ms)':>10} | Description\n"
        )
        sf.write("-" * 140 + "\n")

    results = {}

    for variant_name, variant_conf in SCALING_VARIANTS.items():
        print(f"\n{'='*70}")
        print(f"  Scaling Experiment: {variant_name}  (width_mult = {variant_conf['width_mult']})")
        print(f"  Description: {variant_conf['description']}")
        print(f"{'='*70}")

        # ── Build YAML config ──
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["model_conf"]["width_mult"] = variant_conf["width_mult"]
        cfg["train_conf"]["save_model_dir"] = f"saved_models/scaling/{variant_name}"

        yaml_path = configs_dir / f"{variant_name}_config.yml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"  Config → {yaml_path}")

        # ── Run training ──
        command = [sys.executable, "train.py", "-c", str(yaml_path)]
        print(f"  Launching: {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
            print(f"  ✅ {variant_name} training finished.")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ {variant_name} FAILED: {e}")
            results[variant_name] = {"width_mult": variant_conf["width_mult"], "error": str(e)}
            continue

        # ── Parse results ──
        res = parse_results(Path(cfg["train_conf"]["save_model_dir"]) / "results.txt")
        res["width_mult"] = variant_conf["width_mult"]

        # Convert raw params/flops to M (million)
        if res.get("params") and res["params"] > 0:
            res["params_m"] = round(res["params"] / 1e6, 4)
        if res.get("flops") and res["flops"] > 0:
            res["flops_m"] = round(res["flops"] / 1e6, 4)

        # ── Model file size ──
        size_info = model_file_size(variant_name)
        res["size_kb"] = size_info["size_kb"]

        # ── CPU latency ──
        print(f"\n  Measuring CPU inference latency for {variant_name} …")
        cpu_info = measure_cpu_latency(variant_name, variant_conf["width_mult"])
        res["cpu_mean_ms"] = cpu_info["mean_ms"]
        res["cpu_std_ms"]  = cpu_info["std_ms"]

        results[variant_name] = res

        # ── Append to summary ──
        with open(summary_path, "a") as sf:
            sf.write(
                f"{variant_name:<10} | {res['width_mult']:>4.2f} | "
                f"{res.get('val_acc', 'N/A')!s:>7} | {res.get('val_f1', 'N/A')!s:>7} | "
                f"{res.get('test_acc', 'N/A')!s:>8} | {res.get('test_f1', 'N/A')!s:>8} | "
                f"{res.get('params_m', 'N/A')!s:>9} | {res.get('flops_m', 'N/A')!s:>8} | "
                f"{res.get('size_kb', 'N/A')!s:>8} | "
                f"{res.get('cpu_mean_ms', 'N/A')!s:>10} | "
                f"{variant_conf['description']}\n"
            )

    # ── Save raw results as JSON ──
    json_path = Path("scaling_results_raw.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Raw results → {json_path}")

    # ── Plot trade-off figures ──
    print("\n--- Generating trade-off plots ---")
    plot_tradeoffs(results)
    export_tradeoff_plot(results)

    # ── Print final summary ──
    print(f"\n{'='*70}")
    print(f"  ALL SCALING EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"\nSummary saved to: {summary_path}")
    print("\nQuick view:")
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
