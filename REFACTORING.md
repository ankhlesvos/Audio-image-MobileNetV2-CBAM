# DeepShip 4-Class Mainline vs Legacy Merged Experiments

## Project Structure

This project has been split into two independent tracks:

### 🟢 4-Class DeepShip Mainline (for paper)
- **Labels**: 0=Cargo, 1=Passengership, 2=Tanker, 3=Tug
- **Data**: `data/deepship_4class/`
- **K-Fold Splits**: `data/kfold_4class/`
- **Configs**: `configs/deepship_4class/`
- **Models**: `saved_models/deepship_4class/`
- **Results**: `results/deepship_4class/`

### 🟡 Legacy 3-Class Merged (read-only, for reference only)
- **Labels**: 0=Cargo+Tug, 1=Passengership, 2=Tanker
- **Data**: `data/legacy_merged/` (old data/ directory contents)
- **Configs**: `configs/legacy_merged/`
- **Models**: `saved_models/legacy_merged/`
- **Results**: `results/legacy_merged/`

---

## Quick Start (4-Class)

### Step 1: Generate 4-class data
```bash
python prepare_deepship_data_4class.py --version_name deepship_4class --seed 42
```
This creates `data/deepship_4class/` with 4 subdirectories (0/1/2/3) and train/test list files.

### Step 2: Verify data integrity
```bash
python verify_4class_data.py
```
This checks that all 4 classes have data and configs are correct.

### Step 3: Create k-fold splits (optional)
```bash
python create_kfold_splits_4class.py --k 5 --seed 42
```
This creates `data/kfold_4class/` with 5 stratified folds.

### Step 4: Train a model
```bash
# Single model with base config
python train.py -c configs/deepship_4class/base_config.yml

# Run ablation study (all experiments)
python run_experiments_4class.py

# Run k-fold cross-validation
python run_kfold_4class.py
```

### Step 5: Test a model
```bash
# Use the dedicated 4-class test script
python test_4class.py -c configs/deepship_4class/base_config.yml -m saved_models/deepship_4class/base_config/best_model_epoch_N.pth

# Or use the original test.py (auto-detects num_classes)
python test.py -c configs/deepship_4class/base_config.yml -m saved_models/deepship_4class/base_config/best_model_epoch_N.pth
```

### Step 6: Analyze bad cases
```bash
python analyze_bad_cases_4class.py -m saved_models/deepship_4class/M5_Baseline/best_model.pth
```

---

## File Reference

### New 4-Class Scripts
| File | Purpose |
|------|---------|
| `prepare_deepship_data_4class.py` | Data preparation with correct CLASS_MAP (Tug→3) |
| `create_kfold_splits_4class.py` | K-fold splits for 4-class data |
| `run_experiments_4class.py` | Ablation study runner for 4-class |
| `run_kfold_4class.py` | K-fold CV runner for 4-class |
| `test_4class.py` | Dedicated 4-class test script |
| `analyze_bad_cases_4class.py` | Bad case analysis for 4-class |
| `verify_4class_data.py` | Data integrity verification |

### Updated Files
| File | Change |
|------|--------|
| `test.py` | class_names now auto-derived from num_classes (4 or 3) |
| `modules/teacher_model.py` | Default num_classes changed from 3→4 |
| `train_teacher.py` | Default num_classes fallback changed from 3→4 |

### Legacy Scripts (read-only, in scripts/ subdirectory)
| File | Location |
|------|----------|
| `prepare_deepship_data_5s.py` | `scripts/data_prep/` (original root + scripts/) |
| `create_merged_dataset.py` | `scripts/data_prep/` |
| `create_kfold_splits.py` | `scripts/data_prep/` |
| `run_experiments.py` | `scripts/training/` (original root + scripts/) |
| `run_kfold.py` | `scripts/training/` |

---

## Critical Differences: 4-Class vs Legacy 3-Class

| Aspect | 4-Class Mainline | Legacy 3-Class |
|--------|-----------------|----------------|
| Labels | 0=Cargo, 1=Passenger, 2=Tanker, 3=Tug | 0=Cargo+Tug, 1=Passenger, 2=Tanker |
| CLASS_MAP | `{'Tug': 3}` | `{'Tug': 0}` (merged!) |
| num_classes | 4 | 3 |
| pair_penalty | `[[0,3],[3,0]]` (Cargo↔Tug) | `[[2,0]]` (Tanker→Cargo+Tug) |
| Data dirs | 0/, 1/, 2/, 3/ | 0/, 1/, 2/ |
| Confusion matrix | 4×4 | 3×3 |

---

## ⚠️ Important Warnings

1. **NEVER mix 4-class and legacy 3-class configs** — they point to different data paths
2. **NEVER load a 3-class checkpoint for 4-class training** — classifier shape mismatch (3 vs 4)
3. **NEVER cite legacy results as 4-class results** — the label spaces are fundamentally different
4. **Always run `verify_4class_data.py`** before starting 4-class training
5. **All paper results must come from `saved_models/deepship_4class/` and `results/deepship_4class/`**

---

## Migration History

- **Before**: Cargo(0) and Tug(3) were merged into a single class 0, resulting in 3-class experiments
- **After**: Fully separated into independent 4-class mainline and legacy 3-class reference
- **Key change**: `CLASS_MAP` in data preparation script changed from `{'Tug': 0}` to `{'Tug': 3}`
