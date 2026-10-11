# Best PulseFormer models on egoPPG-DB (seed 0)

This folder contains the **5 best PulseFormer models**, one per fold of the 5-fold cross-validation
on egoPPG-DB, trained with **seed 0**.

```
models/
├── fold_0/cvkfold5_0_Epoch51.pth
├── fold_1/cvkfold5_1_Epoch56.pth
├── fold_2/cvkfold5_2_Epoch34.pth
├── fold_3/cvkfold5_3_Epoch69.pth
└── fold_4/cvkfold5_4_Epoch51.pth
```

The `.pth` checkpoints (~51 MB each) are excluded from git via `.gitignore`. Only this README is
versioned.

## Training setup

- **Config:** `configs/ml/egoppg_egoppg_PulseFormer.yaml`
- **Inputs:** eye-tracking video (`et`, 128×48) + right IMU magnitude (`imu_right`)
- **Label:** nose PPG (`ppg_nose`), `DiffStandardized` format
- **Chunks:** 128 frames (~4.3 s at 30 fps)
- **Training:** 100 epochs, LR 9e-4 (OneCycle), batch size 4, MSE loss
- **Splits:** `KFold(5)` without shuffling over the 25 participants. From the 20 non-test
  participants, 2 are held out for validation (`train_test_split(test_size=2, random_state=0)`),
  which gives 18 train, 2 validation and 5 test participants per fold.
- **Model selection:** a checkpoint is saved every epoch. The epoch with the **lowest validation
  loss** is kept.

## Evaluation

1. The predicted PPG of each test participant is concatenated and split into non-overlapping
   **60 s windows**.
2. Each window is integrated (`cumsum`), detrended, inverted and band-pass filtered
   (Butterworth, 0.7–2.8 Hz).
3. Peaks are detected and HR is computed as `60 / mean IBI`, giving **one HR per window**.
4. The **ground-truth HR** is computed the same way from the reference BVP/PPG signal (`ppg_nose`).

### Metrics

- **MAE, RMSE, MAPE:** error between estimated HR and ground-truth HR over all 60 s windows of the
  5 test participants.
- **Pearson r:** correlation between the **estimated HR** (from the PPG predicted by the network)
  and the **ground-truth HR** (from the reference BVP). It is computed per participant over their
  60 s windows and then averaged over the 5 test participants of the fold. It measures whether the
  estimated HR follows the rises and drops of the true HR, not whether the absolute values match
  (that is what MAE measures).

## Results per fold

### Fold 0

- Model: `fold_0/cvkfold5_0_Epoch51.pth`
- Test: 001–005 · Validation: 007, 024
- Best epoch: **51** (val loss = 1.7022)

| MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson r |
|---|---|---|---|
| 8.03 | 11.16 | 10.20 | 0.89 |

### Fold 1

- Model: `fold_1/cvkfold5_1_Epoch56.pth`
- Test: 006–010 · Validation: 002, 024
- Best epoch: **56** (val loss = 1.5852)

| MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson r |
|---|---|---|---|
| 6.95 | 9.30 | 8.07 | 0.79 |

### Fold 2

- Model: `fold_2/cvkfold5_2_Epoch34.pth`
- Test: 011–015 · Validation: 002, 024
- Best epoch: **34** (val loss = 1.5927)

| MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson r |
|---|---|---|---|
| 8.20 | 10.84 | 9.65 | 0.85 |

### Fold 3

- Model: `fold_3/cvkfold5_3_Epoch69.pth`
- Test: 016–020 · Validation: 002, 024
- Best epoch: **69** (val loss = 1.5174)

| MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson r |
|---|---|---|---|
| 9.17 | 11.80 | 11.14 | 0.86 |

### Fold 4

- Model: `fold_4/cvkfold5_4_Epoch51.pth`
- Test: 021–025 · Validation: 002, 019
- Best epoch: **51** (val loss = 1.5702)

| MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson r |
|---|---|---|---|
| 8.50 | 11.72 | 11.29 | 0.84 |

## Summary

| Fold | Test participants | Best epoch | MAE (bpm) | Pearson r |
|---|---|---|---|---|
| 0 | 001–005 | 51 | 8.03 | 0.89 |
| 1 | 006–010 | 56 | 6.95 | 0.79 |
| 2 | 011–015 | 34 | 8.20 | 0.85 |
| 3 | 016–020 | 69 | 9.17 | 0.86 |
| 4 | 021–025 | 51 | 8.50 | 0.84 |
| **Mean ± std** | | | **8.17 ± 0.72** | **0.85 ± 0.03** |

The std is across folds (`np.std`, as in `main_ml.py`).

Metrics source:
`training_results/runs/testing/egoppg/inputs_et_imu_right/labels_ppg_nose/CL128_Down1_W128_H48_LabelDiffStandardized_VideoTypeDiffStandardized/PulseFormer/seed_0/cvkfold5_<fold>_Evaluation.txt`
