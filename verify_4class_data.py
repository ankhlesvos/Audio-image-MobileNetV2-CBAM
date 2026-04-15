"""
verify_4class_data.py

Verification script for 4-Class DeepShip data integrity.
Run this AFTER prepare_deepship_data_4class.py and BEFORE any training.

Checks:
  1. 4 subdirectories exist (0, 1, 2, 3)
  2. Each subdirectory has WAV files (non-empty)
  3. train_list.txt contains all 4 labels
  4. test_list.txt contains all 4 labels
  5. Class distribution is reasonable (no zero-count classes)
  6. No merged-label artifacts (label '0' should only be Cargo, not Tug)
"""

import os
import sys
from collections import Counter
from pathlib import Path


def verify_data(data_dir="data/deepship_4class", verbose=True):
    data_path = Path(data_dir)
    errors = []
    warnings = []

    print(f"{'=' * 60}")
    print(f"4-Class DeepShip Data Verification")
    print(f"{'=' * 60}")
    print(f"Data directory: {data_path}")
    print()

    # Check 1: 4 subdirectories exist
    print("[Check 1] Verifying 4 class subdirectories...")
    for cid in range(4):
        subdir = data_path / str(cid)
        if not subdir.exists():
            errors.append(f"Missing subdirectory: {subdir}")
            print(f"  [FAIL] Class {cid}: {subdir} does NOT exist")
        elif not subdir.is_dir():
            errors.append(f"Not a directory: {subdir}")
            print(f"  [FAIL] Class {cid}: {subdir} is not a directory")
        else:
            wav_files = list(subdir.glob("*.wav"))
            if len(wav_files) == 0:
                errors.append(f"Empty subdirectory: {subdir}")
                print(f"  [FAIL] Class {cid}: {subdir} has NO WAV files")
            else:
                print(f"  [OK]   Class {cid}: {len(wav_files)} WAV files in {subdir}")

    # Check 2: train_list.txt exists and has all 4 labels
    print("\n[Check 2] Verifying train_list.txt...")
    train_list = data_path / "train_list.txt"
    if not train_list.exists():
        errors.append(f"Missing train list: {train_list}")
        print(f"  [FAIL] {train_list} does NOT exist")
    else:
        train_labels = Counter()
        train_lines = 0
        with open(train_list, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    try:
                        label = int(parts[1])
                        train_labels[label] += 1
                        train_lines += 1
                    except ValueError:
                        errors.append(f"Invalid label in train_list: {line}")

        print(f"  Total lines: {train_lines}")
        for cid in range(4):
            count = train_labels.get(cid, 0)
            if count == 0:
                errors.append(f"Class {cid} has ZERO samples in train_list")
                print(f"  [FAIL] Class {cid}: 0 samples")
            else:
                pct = count / train_lines * 100 if train_lines > 0 else 0
                print(f"  [OK]   Class {cid}: {count} samples ({pct:.1f}%)")

    # Check 3: test_list.txt exists and has all 4 labels
    print("\n[Check 3] Verifying test_list.txt...")
    test_list = data_path / "test_list.txt"
    if not test_list.exists():
        errors.append(f"Missing test list: {test_list}")
        print(f"  [FAIL] {test_list} does NOT exist")
    else:
        test_labels = Counter()
        test_lines = 0
        with open(test_list, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    try:
                        label = int(parts[1])
                        test_labels[label] += 1
                        test_lines += 1
                    except ValueError:
                        errors.append(f"Invalid label in test_list: {line}")

        print(f"  Total lines: {test_lines}")
        for cid in range(4):
            count = test_labels.get(cid, 0)
            if count == 0:
                errors.append(f"Class {cid} has ZERO samples in test_list")
                print(f"  [FAIL] Class {cid}: 0 samples")
            else:
                pct = count / test_lines * 100 if test_lines > 0 else 0
                print(f"  [OK]   Class {cid}: {count} samples ({pct:.1f}%)")

    # Check 4: No merged-label artifacts
    print("\n[Check 4] Checking for merged-label artifacts...")
    # Verify that in the source code, CLASS_MAP has Tug→3, not Tug→0
    prep_script = Path("prepare_deepship_data_4class.py")
    if prep_script.exists():
        with open(prep_script, 'r') as f:
            content = f.read()
            # Look for Tug mapped to 3 (various quote styles)
            if "'Tug': 3" in content or "\"Tug\": 3" in content or "'Tug':3" in content or "'Tug' : 3" in content:
                print(f"  [OK]   {prep_script} has correct CLASS_MAP (Tug→3)")
            elif "'Tug': 0" in content or "\"Tug\": 0" in content or "'Tug':0" in content:
                errors.append(f"{prep_script} still has Tug→0 mapping!")
                print(f"  [FAIL] {prep_script} STILL has Tug→0 mapping!")
            elif "Tug" in content and "3" in content:
                # Heuristic: if both Tug and 3 appear near each other in CLASS_MAP context
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'Tug' in line and 'CLASS_MAP' in ''.join(lines[max(0,i-5):i+5]):
                        if '3' in line and '0' not in line.replace('3', ''):
                            print(f"  [OK]   {prep_script} has correct CLASS_MAP (Tug→3)")
                            break
                else:
                    warnings.append(f"Could not definitively verify CLASS_MAP in {prep_script}")
                    print(f"  [WARN] Could not definitively verify CLASS_MAP in {prep_script}")
            else:
                warnings.append(f"Could not verify CLASS_MAP in {prep_script}")
                print(f"  [WARN] Could not verify CLASS_MAP in {prep_script}")
    else:
        warnings.append(f"{prep_script} not found")
        print(f"  [WARN] {prep_script} not found")

    # Check 5: Config files have num_classes=4
    print("\n[Check 5] Checking 4-class configs for num_classes=4...")
    config_dir = Path("configs/deepship_4class")
    if config_dir.exists():
        yaml_files = list(config_dir.glob("**/*.yml")) + list(config_dir.glob("**/*.yaml"))
        wrong_configs = []
        for yf in yaml_files:
            import yaml
            with open(yf, 'r') as f:
                try:
                    cfg = yaml.safe_load(f)
                    nc = cfg.get('model_conf', {}).get('num_classes', None)
                    if nc is not None and nc != 4:
                        wrong_configs.append((yf, nc))
                        print(f"  [FAIL] {yf}: num_classes={nc} (expected 4)")
                    elif nc == 4:
                        print(f"  [OK]   {yf}: num_classes=4")
                except Exception as e:
                    print(f"  [WARN] {yf}: could not parse ({e})")
        if wrong_configs:
            errors.append(f"Configs with wrong num_classes: {wrong_configs}")
    else:
        warnings.append(f"Config directory {config_dir} not found")
        print(f"  [WARN] {config_dir} not found")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Verification Summary")
    print(f"{'=' * 60}")
    if errors:
        print(f"\n[ERRORS] {len(errors)} error(s) found:")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print(f"\n[OK] No errors found!")

    if warnings:
        print(f"\n[WARNINGS] {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ! {w}")

    if errors:
        print(f"\n*** DATA VERIFICATION FAILED ***")
        print(f"Fix the errors above before running 4-class training.")
        return False
    else:
        print(f"\n*** DATA VERIFICATION PASSED ***")
        print(f"4-class DeepShip data is ready for training.")
        return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/deepship_4class", help="Data directory to verify")
    args = parser.parse_args()

    success = verify_data(args.data_dir)
    sys.exit(0 if success else 1)
