# EVAL_TRAINING: test and evaluation of PulseFormer on egoPPG-DB

This branch (`eval-ppg`) is for testing and evaluating the best PulseFormer models (`models/`, seed 0, one per
fold) on egoPPG-DB. It uses the same evaluation windows as the paper.

For every fold there are two folders of results:

- **`ppg/`**: the BVP (pulse wave) estimated by PulseFormer compared with the nose PPG sensor.
  - One figure per test participant, with one 60 s window per activity.
  - `bvp_pearson_per_participant.csv`: Pearson between the two BVP waves, per participant and activity.
  - `bvp_pearson_summary.csv`: the same Pearson for the whole fold, overall and per activity.
- **`hr/`**: the HR estimated from 60 s windows.
  - One figure per test participant comparing the HR from PulseFormer, the nose PPG sensor (ground truth) and the
    ECG.
  - `hr_metrics_per_participant.csv`: MAE, RMSE, MAPE and Pearson of PulseFormer vs the ground truth, per
    participant and activity.
  - `hr_metrics.csv`: the same metrics for the whole fold, overall and per activity.

At the root of `results/` there are the summaries over the 5 folds: `bvp_summary.csv`, `hr_summary.csv` and
`comparison.csv`.

```
EVAL_TRAINING/
├── ppg_windows.py              # predictions (GPU) + BVP figures for one fold
├── bvp_pearson.py              # BVP Pearson CSVs (per fold) + bvp_summary.csv
├── hr_ecg.py                   # HR figures + HR metrics CSVs (per fold) + hr_summary.csv
├── comparison.py               # comparison.csv
├── run_ppg_windows.sh          # SLURM array job, one task per fold (GPU, gitignored)
├── logs/                       # SLURM logs (gitignored)
└── results/
    ├── fold_<k>/
    │   ├── predictions/<id>.npy                 # [prediction, label] per test participant (gitignored)
    │   ├── ppg/
    │   │   ├── <id>_ppg_windows.png             # BVP figure per test participant
    │   │   ├── bvp_pearson_per_participant.csv  # BVP Pearson per participant and activity
    │   │   └── bvp_pearson_summary.csv          # BVP Pearson of the fold, overall and per activity
    │   └── hr/
    │       ├── <id>_hr.png                      # HR figure per test participant
    │       ├── hr_metrics_per_participant.csv   # HR metrics per participant and activity
    │       └── hr_metrics.csv                   # HR metrics of the fold, overall and per activity
    ├── bvp_summary.csv                          # mean ± std over folds of bvp_pearson_summary.csv
    ├── hr_summary.csv                           # mean ± std over folds of hr_metrics.csv
    └── comparison.csv                           # overall HR metrics and BVP Pearson per fold
```

## How to run

From `egoPPG-tricam/`:

```bash
sbatch EVAL_TRAINING/run_ppg_windows.sh          # predictions + BVP figures, all 5 folds on GPU (~1 min per fold)
python -m EVAL_TRAINING.ppg_windows --fold 1     # one fold; runs on CPU if the predictions already exist
python -m EVAL_TRAINING.bvp_pearson              # BVP Pearson CSVs (CPU, seconds)
python -m EVAL_TRAINING.hr_ecg                   # HR figures + metrics (CPU, ~30 s)
python -m EVAL_TRAINING.comparison               # comparison.csv (after bvp_pearson and hr_ecg)
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

The predictions reproduce the official metrics of `models/README.md` (see `comparison.csv`). The ±0.01 differences
come from GPU non-determinism.

## How the windows are built (as in the paper)

This follows `evaluation/metrics_ppg.py`.

1. **Concatenation.** All kept chunks of 128 frames of a participant are concatenated in recording order (all
   activities, minus the excluded ones), giving one long signal. For example, participant 006 has
   460 chunks × 128 = 58,880 frames = 32.7 min at 30 fps.
2. **60 s windows.** The signal is cut into consecutive, **non-overlapping** (not sliding) windows of 60 s
   (1,800 frames). The last incomplete window is dropped. For example, participant 006 has 32 windows.
3. **BVP and HR per window.** In each window, both the sensor wave and the estimated wave are reconstructed:
   - `cumsum`, to undo the `DiffStandardized` derivative
   - `detrend`
   - inversion
   - Butterworth band-pass filter, 0.7–2.8 Hz

   Then peaks are detected, and `HR = 60 / mean inter-beat interval`. This gives **one sensor HR and one model HR per
   window**.

The windows are cut every 60 s without looking at the activities, so **a window can contain the end of one activity
and the start of the next one** (mixed window). The paper assigns each window to the activity of its **centre
frame**. In fold 1, 31 of the 151 windows (~20 %) are mixed. This happens mostly at activity changes and with the
short walking segments (30–107 s).

The activity of each frame comes from `task_times` in `configs/preprocessing/config_preprocessing_egoppg.yml` (start
and end frame of every activity), with the same chunk counting as `utils.get_task_chunk_list`. The number of frames
was checked against the loaded test chunks for all 25 participants.

---

## 1. `ppg/`: BVP

### Figures (`<id>_ppg_windows.png`)

One figure per test participant, with **one 60 s window per activity**. Only pure windows are used: for each
activity, the figure shows the first window that contains only that activity.

- **Black:** nose PPG sensor (ground truth). **Red:** PulseFormer estimate (from eye video + IMU only).
  Both waves are z-scored for plotting, which does not change the HR or the correlation.
- **▼:** detected peaks, with the same peak detection used to compute the HR.
- **Title:**
  - activity and window index (position of the window in the concatenated signal)
  - HR sensor, HR model and absolute error (bpm)
  - Pearson r between the two waves in that window

An activity is missing from a figure when:
- it is excluded for that participant in the dataset (`exclusion_list`, e.g. bike for 009, 012, 014, 015, 016, 023)
- it has no pure window. This mostly affects walking: only 7 of 25 participants have a pure walking window.

### BVP Pearson

Pearson r between the sensor wave and the estimated wave, **sample by sample**. It measures whether the estimated wave
has the **same shape and timing** as the real one (peaks at the same moments), not only the same rate.

For the CSVs it is computed over **all samples**, not only the windows of the figures:

1. Every 60 s window is reconstructed, both sensor and model (cumsum, detrend, inversion, band-pass, z-score per
   window). The reconstruction is done per window because `cumsum` and `detrend` over a 30 min signal drift too much.
2. Each sample (frame) keeps the activity of its own frame, so mixed windows also count: each part goes to its own
   activity.
3. Pearson r is computed on the pooled samples of each group.

| File | One row per | Samples used |
|---|---|---|
| `bvp_pearson_per_participant.csv` | participant × activity | all samples of that participant in that activity |
| `bvp_pearson_summary.csv` | group: `overall` + each activity | all samples of the 5 test participants of the fold (`overall`: all activities) |

Columns: `pearson_r`, `n_samples` (number of samples, i.e. frames at 30 fps), `minutes` (= `n_samples` / 1,800).

---

## 2. `hr/`: HR

### Three HR sources per window

All 60 s windows are used, including mixed ones, exactly as in the official metric.

| Source | How the HR of a window is computed |
|---|---|
| **PulseFormer** | Estimated wave, reconstructed + peak detection (paper's evaluation) |
| **PPG nose (sensor)** | Sensor wave, same processing. This is the ground truth used in the paper |
| **ECG** | Chest-strap ECG at its original 1024 Hz (`EgoPPG_Dataset/<id>/<id>_ecg_original_fps.npy`): R peaks with neurokit2, `HR = 60 / mean RR interval` |

The ECG is aligned with the eye-tracking frames (exactly 1024/30 ECG samples per frame), so each window uses the ECG
of the same frames. RR intervals across a gap between two activities, and RR intervals outside 0.3–2 s
(30–200 bpm, missed or extra beats), are dropped.

### Figures (`<id>_hr.png`)

HR per window over time for the 3 sources (red: PulseFormer, blue: PPG nose, black: ECG), with the background
coloured by activity (activity of the centre frame). The title gives the MAE of PulseFormer vs PPG nose for that
participant.

### HR metrics

**PulseFormer vs PPG nose (sensor)**, the paper's metric, with the same definitions as `evaluation/metrics_ppg.py`.
The ECG is only shown in the figures. Windows are assigned to activities by their centre frame.

| File | One row per | Windows used |
|---|---|---|
| `hr_metrics_per_participant.csv` | participant × group (`overall` + each activity) | windows of that participant |
| `hr_metrics.csv` | group (`overall` + each activity) | windows of the 5 test participants of the fold |

| Column | Meaning |
|---|---|
| `n_windows` | Number of 60 s windows |
| `MAE`, `RMSE` | bpm, over the windows |
| `MAPE` | %, relative to the PPG nose HR |
| `Pearson` | Correlation of the two HR series over the windows. In `hr_metrics.csv`: per participant, then averaged over participants (as in the paper) |

In `hr_metrics_per_participant.csv`, Pearson needs at least 3 windows (with 2 windows it is always ±1) and is
otherwise empty. With only 3–5 windows per activity it is very noisy.

---

## 3. Summaries over the 5 folds (root of `results/`)

`bvp_summary.csv` and `hr_summary.csv` are the **mean and std over the 5 folds** of the per-fold summaries, for
`overall` and for each activity. This is how the paper reports its results: one value per fold, then mean ± std.

### `bvp_summary.csv` (mean ± std over folds of `ppg/bvp_pearson_summary.csv`)

| Group | BVP Pearson | Minutes (5 folds) |
|---|---|---|
| **overall** | **0.22 ± 0.07** | 714 |
| video | 0.52 ± 0.11 | 116 |
| office | 0.29 ± 0.10 | 231 |
| walking | 0.08 ± 0.03 | 45 |
| kitchen | 0.10 ± 0.06 | 127 |
| dancing | −0.01 ± 0.03 | 118 |
| bike | 0.12 ± 0.06 | 78 |

### `hr_summary.csv` (mean ± std over folds of `hr/hr_metrics.csv`)

| Group | Windows (5 folds) | MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson |
|---|---|---|---|---|---|
| **overall** | 714 | **8.17 ± 0.72** | 10.96 ± 0.90 | 10.07 ± 1.17 | 0.85 ± 0.03 |
| video | 119 | 5.30 ± 1.03 | 7.84 ± 1.89 | 8.29 ± 2.30 | 0.47 ± 0.10 |
| office | 226 | 7.46 ± 1.18 | 10.12 ± 1.91 | 10.72 ± 2.26 | 0.50 ± 0.12 |
| walking | 45 | 9.08 ± 2.39 | 10.67 ± 2.70 | 9.06 ± 2.85 | 0.56 ± 0.37 |
| kitchen | 129 | 8.96 ± 1.23 | 11.38 ± 2.25 | 10.96 ± 1.96 | 0.45 ± 0.12 |
| dancing | 116 | 9.65 ± 3.01 | 11.74 ± 3.29 | 10.97 ± 3.51 | 0.39 ± 0.22 |
| bike | 79 | 11.77 ± 2.47 | 14.51 ± 2.91 | 9.74 ± 2.05 | 0.46 ± 0.22 |

The overall row matches `models/README.md` (MAE 8.17 ± 0.72, Pearson 0.85 ± 0.03).

### `comparison.csv` (overall metrics per fold)

The `overall` row of every fold, HR metrics (`hr/hr_metrics.csv`) next to the BVP Pearson
(`ppg/bvp_pearson_summary.csv`), plus mean and std over folds:

| Fold | Windows | HR MAE (bpm) | HR RMSE (bpm) | HR MAPE (%) | HR Pearson | BVP Pearson |
|---|---|---|---|---|---|---|
| 0 | 153 | 8.04 | 11.16 | 10.21 | 0.89 | 0.13 |
| 1 | 151 | 6.94 | 9.30 | 8.07 | 0.79 | 0.17 |
| 2 | 120 | 8.20 | 10.84 | 9.65 | 0.85 | 0.31 |
| 3 | 149 | 9.16 | 11.79 | 11.13 | 0.86 | 0.27 |
| 4 | 141 | 8.49 | 11.71 | 11.29 | 0.84 | 0.21 |
| **mean ± std** | | **8.17 ± 0.72** | **10.96 ± 0.90** | **10.07 ± 1.17** | **0.85 ± 0.04** | **0.22 ± 0.06** |

The two Pearson values measure different things:
- **HR Pearson (0.85):** whether the HR per minute goes up and down with the real HR.
- **BVP Pearson (0.22):** whether the wave itself matches the real one, sample by sample.

PulseFormer tracks the HR trend well, but recovers little of the actual pulse waveform. The waveform is recovered
best at rest (video, BVP r ≈ 0.5) and is essentially lost with motion (r ≈ 0).
