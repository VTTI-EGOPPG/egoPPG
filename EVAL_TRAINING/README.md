# EVAL_TRAINING: BVP and HR evaluation of PulseFormer

Evaluation of the best PulseFormer models (`models/`) on egoPPG-DB, using the same evaluation windows as the paper:

1. **BVP:** waves estimated by PulseFormer compared with the nose PPG sensor (figures + wave correlation).
2. **HR:** HR per 60 s window from PulseFormer, the nose PPG sensor and the ECG (figures), and metrics of
   PulseFormer vs the nose PPG sensor.

```
EVAL_TRAINING/
├── ppg_windows.py              # predictions (GPU) + BVP figures for one fold
├── bvp_pearson.py              # pooled BVP correlation, per fold and for all folds
├── hr_ecg.py                   # HR figures (PulseFormer / PPG nose / ECG) + HR metrics, all folds
├── comparison.py               # HR metrics vs BVP correlation, overall per fold
├── run_ppg_windows.sh          # SLURM array job, one task per fold (GPU)
├── logs/                       # SLURM logs (gitignored)
└── results/
    ├── fold_<k>/
    │   ├── predictions/<id>.npy             # [prediction, label] per test participant (gitignored)
    │   ├── ppg/<id>_ppg_windows.png         # BVP figure per test participant
    │   ├── hr/
    │   │   ├── <id>_hr.png                  # HR figure per test participant
    │   │   └── hr_metrics.csv               # HR metrics of the fold (PulseFormer vs PPG nose)
    │   ├── bvp_pearson_summary.csv          # pooled BVP r of the fold: overall and per activity
    │   └── bvp_pearson_per_participant.csv  # pooled BVP r per participant and activity
    ├── bvp_pearson_summary.csv              # same, all 5 folds together
    ├── bvp_pearson_per_participant.csv
    ├── hr_metrics.csv                       # HR metrics, all folds together
    └── comparison.csv                       # HR metrics + BVP Pearson, per fold and all folds
```

## How to run

From `egoPPG-tricam/`:

```bash
sbatch EVAL_TRAINING/run_ppg_windows.sh          # predictions + BVP figures, all 5 folds on GPU (~1 min per fold)
python -m EVAL_TRAINING.ppg_windows --fold 1     # one fold; runs on CPU if the predictions already exist
python -m EVAL_TRAINING.bvp_pearson              # pooled BVP correlations (CPU, seconds)
python -m EVAL_TRAINING.hr_ecg                   # HR figures + metrics (CPU, ~30 s)
python -m EVAL_TRAINING.comparison               # comparison table (after hr_ecg and bvp_pearson)
```

The predictions are only computed when `results/fold_<k>/predictions/` is empty. Everything else runs on CPU from
the saved predictions.

## No data leakage

Each fold has its own model, and each model is evaluated only on the 5 test participants of its own fold, which it
never saw during training or validation. Across the 5 folds, each of the 25 participants is evaluated exactly once.

| Fold | Model | Test participants |
|---|---|---|
| 0 | `models/fold_0/cvkfold5_0_Epoch51.pth` | 001–005 |
| 1 | `models/fold_1/cvkfold5_1_Epoch56.pth` | 006–010 |
| 2 | `models/fold_2/cvkfold5_2_Epoch34.pth` | 011–015 |
| 3 | `models/fold_3/cvkfold5_3_Epoch69.pth` | 016–020 |
| 4 | `models/fold_4/cvkfold5_4_Epoch51.pth` | 021–025 |

The predictions reproduce the official metrics of `models/README.md` (see the HR section). The ±0.01 differences
come from GPU non-determinism.

## How the windows are built (as in the paper)

This follows `evaluation/metrics_ppg.py`.

1. **Concatenation.** All kept chunks of 128 frames of a participant are concatenated in recording order (all
   activities, minus the excluded ones), giving one long signal. For example, participant 006 has
   460 chunks × 128 = 58,880 frames = 32.7 min at 30 fps.
2. **60 s windows.** The signal is cut into consecutive, **non-overlapping** (not sliding) windows of 60 s
   (1,800 frames). The last incomplete window is dropped. For example, participant 006 has 32 windows.
3. **Pulse and HR per window.** In each window, both the sensor wave and the estimated wave are reconstructed:
   - `cumsum`, to undo the `DiffStandardized` derivative
   - `detrend`
   - inversion
   - Butterworth band-pass filter, 0.7–2.8 Hz

   Then peaks are detected, and `HR = 60 / mean inter-beat interval`. This gives **one sensor HR and one model HR per
   window**.

### Mixed windows

The windows are cut every 60 s without looking at the activities, so **a window can contain the end of one activity
and the start of the next one**. The paper assigns each window to the activity of its **centre frame**. In fold 1,
31 of the 151 windows (~20 %) mix two or more activities. This happens mostly at activity changes and with the short
walking segments (30–107 s).

The activity of each frame comes from `task_times` in `configs/preprocessing/config_preprocessing_egoppg.yml` (start
and end frame of every activity), with the same chunk counting as `utils.get_task_chunk_list`. The number of frames
was checked against the loaded test chunks for all 25 participants.

---

## 1. BVP

### Figures (`fold_<k>/ppg/<id>_ppg_windows.png`)

**Only pure windows are used:** for each activity, the figure shows the first window that contains only that
activity, as one panel of 60 s.

- **Black:** nose PPG sensor (ground truth). **Red:** PulseFormer estimate (from eye video + IMU only).
  Both waves are z-scored for plotting, which does not change the HR or the correlation.
- **▼:** detected peaks, with the same peak detection used to compute the HR.
- **Title:**
  - activity and window index (position of the window in the concatenated signal)
  - HR sensor, HR model and absolute error (bpm)
  - Pearson r of that window

An activity is missing from a figure when:
- it is excluded for that participant in the dataset (`exclusion_list`, e.g. bike for 009, 012, 014, 015, 016, 023)
- it has no pure window. This mostly affects walking: only 7 of 25 participants have a pure walking window.

**Pearson r in the figures** is the correlation between the sensor wave and the estimated wave, sample by sample,
inside that one window (1,800 samples, after reconstruction): `np.corrcoef(wave_sensor, wave_model)`. It measures
whether the estimated wave has the **same shape and timing** as the real one (peaks at the same moments), not only the
same rate. A small HR error with r ≈ 0 means that the model oscillates at the right rate, but its beats do not line
up with the real ones.

### Pooled BVP correlation (`bvp_pearson.py`)

Instead of a single window, the correlation is computed over **all BVP samples**:

1. Every 60 s window is reconstructed, both sensor and model (cumsum, detrend, inversion, band-pass, z-score per
   window). The reconstruction is done per window because `cumsum` and `detrend` over a 30 min signal drift too much.
2. Each sample keeps the activity of its own frame, so mixed windows also count: each part goes to its own activity.
3. Pearson r is computed on all pooled samples: overall, per activity, and per participant and activity.

In `bvp_pearson_summary.csv`, `n_samples` is the number of pooled samples and `minutes` = `n_samples` / 1,800.

All 5 folds, 25 participants, 714 min:

| Group | Pearson r | Minutes |
|---|---|---|
| **overall** | **0.21** | 714 |
| video | 0.52 | 116 |
| office | 0.29 | 231 |
| walking | 0.09 | 45 |
| kitchen | 0.10 | 127 |
| dancing | −0.02 | 118 |
| bike | 0.11 | 78 |

Overall per fold: 0.13 (fold 0), 0.17 (fold 1), 0.31 (fold 2), 0.27 (fold 3), 0.21 (fold 4).

At rest (video), the estimated wave follows the sensor. With motion, the wave shape is lost (r ≈ 0).

---

## 2. HR

### Three HR sources per window

All windows of the paper are used, including mixed ones, exactly as in the official metric.

| Source | How the HR of a window is computed |
|---|---|
| **PulseFormer** | Estimated wave, reconstructed + peak detection (paper's evaluation) |
| **PPG nose (sensor)** | Sensor wave, same processing. This is the ground truth used in the paper |
| **ECG** | Chest-strap ECG at its original 1024 Hz (`EgoPPG_Dataset/<id>/<id>_ecg_original_fps.npy`): R peaks with neurokit2, `HR = 60 / mean RR interval` |

The ECG is aligned with the eye-tracking frames (exactly 1024/30 ECG samples per frame), so each window uses the ECG
of the same frames. RR intervals across a gap between two activities, and RR intervals outside 0.3–2 s
(30–200 bpm, missed or extra beats), are dropped.

### Figures (`fold_<k>/hr/<id>_hr.png`)

HR per window over time for the 3 sources (red: PulseFormer, blue: PPG nose, black: ECG), with the background
coloured by activity (activity of the centre frame). The title gives the MAE of PulseFormer vs PPG nose for that
participant.

### Metrics (`hr/hr_metrics.csv`)

**PulseFormer vs PPG nose (sensor)**, the paper's metric, with the same definitions as `evaluation/metrics_ppg.py`.
The ECG is only shown in the figures.

| Column | Meaning |
|---|---|
| `group` | `overall` or one activity (windows assigned by centre frame) |
| `n_windows` | Number of 60 s windows |
| `MAE`, `RMSE` | bpm, over all windows |
| `MAPE` | %, relative to the PPG nose HR |
| `Pearson` | Correlation of the two HR series over the windows of each participant, averaged over participants |

`fold_<k>/hr/hr_metrics.csv` uses the 5 test participants of the fold. `results/hr_metrics.csv` pools all 714
windows of the 25 participants. Its MAE (8.16) is therefore weighted by windows, and differs slightly from the mean
over folds in `models/README.md` (8.17).

Per fold, the overall row matches the official metrics:

| Fold | MAE | RMSE | MAPE | Pearson |
|---|---|---|---|---|
| 0 | 8.04 | 11.16 | 10.21 | 0.89 |
| 1 | 6.94 | 9.30 | 8.07 | 0.79 |
| 2 | 8.20 | 10.84 | 9.65 | 0.85 |
| 3 | 9.16 | 11.79 | 11.13 | 0.87 |
| 4 | 8.49 | 11.71 | 11.29 | 0.84 |

All 5 folds (`results/hr_metrics.csv`):

| Group | Windows | MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson |
|---|---|---|---|---|---|
| **overall** | 714 | **8.16** | 10.99 | 10.07 | 0.85 |
| video | 119 | 5.26 | 7.99 | 8.21 | 0.47 |
| office | 226 | 7.47 | 10.30 | 10.73 | 0.50 |
| walking | 45 | 8.91 | 10.75 | 9.31 | 0.43 |
| kitchen | 129 | 9.01 | 11.66 | 11.01 | 0.45 |
| dancing | 116 | 9.20 | 11.67 | 10.54 | 0.37 |
| bike | 79 | 11.13 | 14.31 | 9.19 | 0.49 |

---

## 3. HR vs BVP (`comparison.csv`)

Overall of PulseFormer vs PPG nose: HR metrics per 60 s window (from `hr_metrics.csv`) next to the pooled BVP Pearson
(from `bvp_pearson_summary.csv`).

| Group | Windows | HR MAE (bpm) | HR RMSE (bpm) | HR MAPE (%) | HR Pearson | BVP Pearson |
|---|---|---|---|---|---|---|
| fold 0 | 153 | 8.04 | 11.16 | 10.21 | 0.89 | 0.13 |
| fold 1 | 151 | 6.94 | 9.30 | 8.07 | 0.79 | 0.17 |
| fold 2 | 120 | 8.20 | 10.84 | 9.65 | 0.85 | 0.31 |
| fold 3 | 149 | 9.16 | 11.79 | 11.13 | 0.86 | 0.27 |
| fold 4 | 141 | 8.49 | 11.71 | 11.29 | 0.84 | 0.21 |
| **mean ± std over folds** | | **8.17 ± 0.72** | **10.96 ± 0.90** | **10.07 ± 1.17** | **0.85 ± 0.04** | **0.22 ± 0.06** |
| all folds pooled | 714 | 8.16 | 10.99 | 10.07 | 0.85 | 0.21 |

The two Pearson values measure different things:
- **HR Pearson (0.85):** whether the HR per minute goes up and down with the real HR.
- **BVP Pearson (0.21):** whether the wave itself matches the real one, sample by sample.

PulseFormer tracks the HR trend well, but recovers little of the actual pulse waveform.
