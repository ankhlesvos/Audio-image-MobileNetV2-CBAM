"""
run_kfold_4class.py

K-fold cross-validation runner for 4-Class DeepShip experiments.

For each (experiment, seed) pair, iterates over K folds,
trains a model with train.py logic, and reports mean ± std
of test file-level macro-F1 and Acc.

Writes:
  results/deepship_4class/kfold_results_summary.csv  — one row per (exp, seed, fold)
  results/deepship_4class/kfold_aggregate.txt        — mean ± std aggregated across folds × seeds

Usage (dry-run 2 epochs, 1 seed, M5_Baseline only):
    python3 run_kfold_4class.py --max_epoch 2 --seeds 42 --exps M5_Baseline

Full run (3 seeds, 5 folds, all experiments):
    python3 run_kfold_4class.py

Labels: 0=Cargo, 1=Passengership, 2=Tanker, 3=Tug
"""

import argparse
import copy
import csv
import os
import sys
import subprocess
from pathlib import Path

import yaml

# ── Shared experiment config template ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from run_experiments_4class import (
    BASE_CONFIG_TEMPLATE,
    CONFIG_BASE, CONFIG_CBAM_S24, CONFIG_SE_ALL, CONFIG_FREQ_S24,
    EXPERIMENTS,
)

DRIVE_BASE = Path(".")

KFOLD_DIR      = Path("data/kfold_4class")
RESULTS_CSV    = Path("results/deepship_4class/kfold_results_summary.csv")
AGGREGATE_TXT  = Path("results/deepship_4class/kfold_aggregate.txt")
CONFIGS_DIR    = Path("configs/deepship_4class/kfold")

# ── Which experiments to run (override via --exps) ────────────────────────────
DEFAULT_EXPS = [
    "M5_Baseline",
    "M5_Plus_Sampler",
    "M5_LA_TT",
]

DEFAULT_SEEDS  = [42, 43, 44]
DEFAULT_K      = 5
DEFAULT_EPOCHS = 50


def make_config(setup: dict, train_list: str, val_list: str,
                save_dir: str, seed: int, max_epoch: int) -> dict:
    """Build a training config dict for one fold/seed/experiment."""
    cfg = copy.deepcopy(BASE_CONFIG_TEMPLATE)
    cfg["train_conf"]["seed"]       = seed
    cfg["train_conf"]["max_epoch"]  = max_epoch
    cfg["train_conf"]["save_model_dir"] = save_dir
    cfg["train_conf"]["patience"]   = setup.get("patience", 10)

    # Model settings
    cfg["model_conf"]["model_config"]       = setup["model_config"]
    cfg["model_conf"]["asymmetric"]         = setup.get("asymmetric", False)
    cfg["model_conf"]["multiscale"]         = setup.get("multiscale", False)
    cfg["model_conf"]["force_no_residual"]  = setup.get("force_no_residual", False)
    cfg["model_conf"]["audio_mode"]         = setup.get("audio_mode", False)
    if "in_channels" in setup:
        cfg["model_conf"]["in_channels"] = setup["in_channels"]

    # Training settings
    cfg["train_conf"]["use_sampler"]        = setup.get("use_sampler", False)
    cfg["train_conf"]["sampler_alpha"]      = setup.get("sampler_alpha", 0.5)
    cfg["train_conf"]["use_class_weights"]  = setup.get("use_class_weights", False)
    if "learning_rate" in setup:
        cfg["train_conf"]["learning_rate"] = setup["learning_rate"]
    if "file_voting_strategy" in setup:
        cfg["train_conf"]["file_voting_strategy"] = setup["file_voting_strategy"]
    if "file_voting_top_k" in setup:
        cfg["train_conf"]["file_voting_top_k"] = setup["file_voting_top_k"]

    # Loss settings
    lc = cfg["train_conf"]["loss_conf"]
    lc["loss_type"]                = setup.get("loss_type", "ce")
    lc["gamma"]                    = setup.get("gamma", 0.0)
    lc["label_smoothing"]          = setup.get("label_smoothing", 0.1)
    lc["apply_logit_adj_in_train"] = setup.get("apply_logit_adj_in_train", False)
    lc["apply_logit_adj_in_eval"]  = setup.get("apply_logit_adj_in_eval", False)
    lc["logit_adj_tau"]            = setup.get("logit_adj_tau", 1.0)

    # Data paths (fold-specific)
    cfg["data_conf"]["train_list"] = train_list
    cfg["data_conf"]["val_list"]   = val_list
    cfg["data_conf"]["test_list"]  = "data/deepship_4class/test_list.txt"

    return cfg


def parse_results(results_path: Path) -> dict:
    """Parse results.txt → dict of metric name → float value."""
    metrics = {}
    if not results_path.exists():
        return metrics
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, _, val = line.partition(":")
                try:
                    metrics[key.strip()] = float(val.strip().split()[0])
                except (ValueError, IndexError):
                    pass
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",          type=int, default=DEFAULT_K)
    parser.add_argument("--seeds",      type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--exps",       nargs="+", default=DEFAULT_EXPS)
    parser.add_argument("--max_epoch",  type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()

    K     = args.k
    seeds = args.seeds
    exps  = args.exps

    DRIVE_BASE.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Verify fold files exist ────────────────────────────────────────────────
    for fold in range(K):
        for suffix in ("train", "val"):
            p = KFOLD_DIR / f"fold{fold}_{suffix}.txt"
            if not p.exists():
                print(f"[ERROR] Missing fold file: {p}")
                print("        Run: python3 create_kfold_splits_4class.py first.")
                sys.exit(1)

    # ── Verify 4-class data exists ─────────────────────────────────────────────
    test_list = Path("data/deepship_4class/test_list.txt")
    if not test_list.exists():
        print(f"[ERROR] Missing 4-class test list: {test_list}")
        print("        Run: python3 prepare_deepship_data_4class.py first.")
        sys.exit(1)

    # ── Ensure results directory exists ────────────────────────────────────────
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ── Init CSV ───────────────────────────────────────────────────────────────
    csv_fields = ["exp_name", "seed", "fold",
                  "val_f1", "val_acc", "test_file_f1", "test_file_acc",
                  "test_seg_f1", "test_seg_acc"]

    with open(RESULTS_CSV, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_fields)
        writer.writeheader()

    # ── Main loop ─────────────────────────────────────────────────────────────
    all_rows: list[dict] = []

    for exp_name in exps:
        if exp_name not in EXPERIMENTS:
            print(f"[SKIP] Unknown experiment: {exp_name}")
            continue
        setup = EXPERIMENTS[exp_name]

        for seed in seeds:
            for fold in range(K):
                tag = f"{exp_name}_seed{seed}_fold{fold}"
                print(f"\n{'=' * 20} {tag} (4-CLASS) {'=' * 20}")

                train_list = str(KFOLD_DIR / f"fold{fold}_train.txt")
                val_list   = str(KFOLD_DIR / f"fold{fold}_val.txt")
                save_dir   = f"saved_models/deepship_4class/kfold/{tag}"

                cfg = make_config(setup, train_list, val_list,
                                  save_dir, seed, args.max_epoch)

                config_path = CONFIGS_DIR / f"{tag}_config.yml"
                with open(config_path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False)

                cmd = [sys.executable, "train.py", "-c", str(config_path)]
                print(f"  Running: {' '.join(cmd)}")

                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"  [FAILED] {e}")
                    continue

                # Parse results
                results_path = Path(save_dir) / "results.txt"
                m = parse_results(results_path)

                row = {
                    "exp_name":      exp_name,
                    "seed":          seed,
                    "fold":          fold,
                    "val_f1":        m.get("Best Val F1",  float("nan")),
                    "val_acc":       m.get("Best Val Acc", float("nan")),
                    "test_file_f1":  m.get("Test File F1", float("nan")),
                    "test_file_acc": m.get("Test File Acc", float("nan")),
                    "test_seg_f1":   m.get("Test Seg  F1",  float("nan")),
                    "test_seg_acc":  m.get("Test Seg  Acc", float("nan")),
                }
                all_rows.append(row)

                with open(RESULTS_CSV, "a", newline="") as csvf:
                    writer = csv.DictWriter(csvf, fieldnames=csv_fields)
                    writer.writerow(row)

                print(f"  Results: test_file_f1={row['test_file_f1']:.4f}  "
                      f"test_file_acc={row['test_file_acc']:.4f}")

    # ── Aggregate: mean ± std per experiment (across folds × seeds) ───────────
    import statistics

    agg: dict[str, dict[str, list]] = {}
    for row in all_rows:
        key = row["exp_name"]
        if key not in agg:
            agg[key] = {f: [] for f in csv_fields[3:]}
        for field in csv_fields[3:]:
            v = row[field]
            if v == v:  # nan check
                agg[key][field].append(v)

    lines_out = [
        "4-Class DeepShip K-Fold Cross-Validation Aggregate Results",
        f"K={K}  seeds={seeds}  experiments={exps}",
        "=" * 100,
        f"{'Experiment':<30}  {'Test File F1':>14}  {'Test File Acc':>14}  "
        f"{'Val F1':>10}  {'N':>4}",
        "-" * 100,
    ]

    for exp_name in exps:
        if exp_name not in agg:
            continue
        vals = agg[exp_name]

        def fmt(lst):
            if len(lst) >= 2:
                return f"{statistics.mean(lst):.4f} ± {statistics.stdev(lst):.4f}"
            elif lst:
                return f"{lst[0]:.4f} ±    N/A"
            return "N/A"

        n    = len(vals["test_file_f1"])
        line = (f"{exp_name:<30}  {fmt(vals['test_file_f1']):>14}  "
                f"{fmt(vals['test_file_acc']):>14}  "
                f"{fmt(vals['val_f1']):>14}  {n:>4}")
        lines_out.append(line)

    lines_out.append("=" * 100)
    aggregate_text = "\n".join(lines_out) + "\n"

    with open(AGGREGATE_TXT, "w") as f:
        f.write(aggregate_text)

    print(f"\n\n{aggregate_text}")
    print(f"Wrote {RESULTS_CSV}  and  {AGGREGATE_TXT}")


if __name__ == "__main__":
    main()
