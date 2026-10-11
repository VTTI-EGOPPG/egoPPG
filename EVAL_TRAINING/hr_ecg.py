"""HR per 60 s window from PulseFormer, the nose PPG sensor and the ECG, plus metrics.

Uses the predictions saved by EVAL_TRAINING/ppg_windows.py. The windows are the paper's evaluation windows
(evaluation/metrics_ppg.py): non-overlapping 60 s windows over the concatenated kept chunks, incomplete last window
dropped, each assigned to the activity of its centre frame. In every window:
    - HR PulseFormer and HR PPG nose: reconstructed wave + peak detection, exactly as in the paper's evaluation
    - HR ECG: R peaks of the chest-strap ECG at its original 1024 Hz (neurokit2), HR = 60 / mean RR interval.
      The ECG is cut to the same frames as the window (frame f = ECG samples f * 1024 / 30). RR intervals across a
      gap between two activities, and RR intervals outside 0.3-2 s (30-200 bpm, missed or extra beats), are dropped.

Metrics of PulseFormer vs PPG nose (the paper's metric: MAE, RMSE, MAPE, Pearson), computed as in
evaluation/metrics_ppg.py: MAE/RMSE/MAPE over all windows, Pearson per participant over its windows and then averaged
over participants. Written to results/fold_<k>/hr/hr_metrics.csv and results/hr_metrics.csv (all folds).

Usage (from egoPPG-tricam/, after ppg_windows.py has saved the predictions):
    python -m EVAL_TRAINING.hr_ecg
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import yaml

from evaluation.post_process import calculate_metric_per_video_ppg
from EVAL_TRAINING.ppg_windows import CFG_PATH, CFG_PRE_PATH, FS, OUT_ROOT, TASK_COLORS, WINDOW
from ml.config import get_config
from utils import get_adjusted_task_times

DATA_ROOT = '/projects/aria_rppg/EgoPPG_Dataset'
FS_ECG = 1024
ACTIVITY_ORDER = ['video', 'office', 'walking', 'kitchen', 'dancing', 'bike']
SOURCES = {'hr_pulseformer': 'PulseFormer', 'hr_ppg_nose': 'PPG nose (sensor)', 'hr_ecg': 'ECG'}
COMPARISONS = [('hr_pulseformer', 'hr_ppg_nose')]  # the paper's metric; the ECG is only plotted


def kept_frames(config, participant):
    """Original frame index (in <id>_et.npy) and activity of every frame of the concatenated prediction.
    Same chunk counting and task selection as utils.get_task_chunk_list."""
    with open(CFG_PRE_PATH) as f:
        cfg_pre = yaml.load(f, Loader=yaml.FullLoader)
    chunk = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
    idx, tasks = [], []
    for task, (t0, t1) in get_adjusted_task_times(cfg_pre['task_times'][participant]).items():
        name = 'walking' if task.startswith('walking') else task
        n_frames = ((t1 - t0) // config.TRAIN.DATA.PREPROCESS.DOWNSAMPLE + 1) // chunk * chunk
        if name in config.TASKS_TO_USE and participant not in cfg_pre['exclusion_list'][name]:
            idx.append(np.arange(t0, t0 + n_frames))
            tasks += [name] * n_frames
    return np.concatenate(idx), np.asarray(tasks)


def ecg_r_peaks(participant):
    ecg = np.load(f'{DATA_ROOT}/{participant}/{participant}_ecg_original_fps.npy')
    _, info = nk.ecg_peaks(nk.ecg_clean(ecg, sampling_rate=FS_ECG), sampling_rate=FS_ECG)
    return np.asarray(info['ECG_R_Peaks'])


def ecg_hr(r_peaks, frame_idx):
    """HR from the R peaks inside the ECG samples of these frames, without RR intervals across activity gaps."""
    rr = []
    breaks = np.r_[0, np.flatnonzero(np.diff(frame_idx) != 1) + 1, len(frame_idx)]
    for s, e in zip(breaks[:-1], breaks[1:]):
        lo, hi = frame_idx[s] * FS_ECG / FS, (frame_idx[e - 1] + 1) * FS_ECG / FS
        peaks = r_peaks[(r_peaks >= lo) & (r_peaks < hi)]
        rr.extend(np.diff(peaks) / FS_ECG)
    rr = np.asarray(rr)
    rr = rr[(rr >= 0.3) & (rr <= 2.0)]
    return 60 / rr.mean() if len(rr) else np.nan


def participant_windows(config, participant, pred, label, r_peaks):
    frame_idx, tasks = kept_frames(config, participant)
    assert len(frame_idx) == len(pred), (participant, len(frame_idx), len(pred))
    rows = []
    for i_window, i in enumerate(range(0, len(pred) - WINDOW + 1, WINDOW)):
        hr_ppg, hr_pred, _, _, _ = calculate_metric_per_video_ppg(pred[i:i + WINDOW], label[i:i + WINDOW],
                                                                  'Peak_Detection', True, FS)
        rows.append({'participant': participant, 'window': i_window, 'activity': tasks[i + WINDOW // 2],
                     'hr_pulseformer': hr_pred, 'hr_ppg_nose': hr_ppg,
                     'hr_ecg': ecg_hr(r_peaks, frame_idx[i:i + WINDOW])})
    return pd.DataFrame(rows)


def plot_participant(df, fold, out_dir):
    participant = df.participant.iloc[0]
    fig, ax = plt.subplots(figsize=(16, 5))
    for _, w in df.iterrows():
        ax.axvspan(w.window, w.window + 1, color=TASK_COLORS[w.activity], alpha=0.12, lw=0)
    minute = df.window + 0.5
    for col, (marker, color) in zip(SOURCES, [('^', 'tab:red'), ('o', 'tab:blue'), ('s', 'black')]):
        ax.plot(minute, df[col], marker=marker, color=color, ms=5, label=SOURCES[col])
    mae = np.nanmean(np.abs(df.hr_pulseformer - df.hr_ppg_nose))
    ax.set_title(f'Participant {participant} (fold {fold}): HR per 60 s window  |  '
                 f'MAE PulseFormer vs PPG nose = {mae:.2f} bpm', fontsize=11)
    ax.set_xlabel('window (minute of the concatenated signal)')
    ax.set_ylabel('HR (bpm)')
    ax.set_xlim(0, df.window.max() + 1)
    activities = [a for a in ACTIVITY_ORDER if a in set(df.activity)]
    handles = ax.get_legend_handles_labels()[0] + [plt.Rectangle((0, 0), 1, 1, color=TASK_COLORS[a], alpha=0.3)
                                                    for a in activities]
    ax.legend(handles, list(SOURCES.values()) + activities, ncol=9, fontsize=8, loc='upper left')
    fig.savefig(f'{out_dir}/{participant}_hr.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


def metrics(df):
    """Rows: comparison x group (overall + activities). Same definitions as evaluation/metrics_ppg.py."""
    rows = []
    groups = [('overall', df)] + [(a, df[df.activity == a]) for a in ACTIVITY_ORDER if a in set(df.activity)]
    for est, ref in COMPARISONS:
        for group, g in groups:
            g = g.dropna(subset=[est, ref])
            err = g[est] - g[ref]
            pearson = [np.corrcoef(p[ref], p[est])[0, 1] for _, p in g.groupby('participant') if len(p) > 1]
            rows.append({'comparison': f'{SOURCES[est]} vs {SOURCES[ref]}', 'group': group, 'n_windows': len(g),
                         'MAE': err.abs().mean(), 'RMSE': np.sqrt((err ** 2).mean()),
                         'MAPE': (err.abs() / g[ref]).mean() * 100, 'Pearson': np.nanmean(pearson)})
    return pd.DataFrame(rows)


def main():
    config = get_config(argparse.Namespace(cfg_path=CFG_PATH))
    all_df = []
    for fold in range(5):
        files = sorted(glob.glob(f'{OUT_ROOT}/fold_{fold}/predictions/*.npy'))
        if not files:
            print(f'Fold {fold}: no predictions found, skipped (run EVAL_TRAINING/ppg_windows.py first)')
            continue
        out_dir = f'{OUT_ROOT}/fold_{fold}/hr'
        os.makedirs(out_dir, exist_ok=True)
        fold_df = []
        for f in files:
            participant = os.path.basename(f).split('.')[0]
            pred, label = np.load(f)
            df = participant_windows(config, participant, pred, label, ecg_r_peaks(participant))
            plot_participant(df, fold, out_dir)
            fold_df.append(df.assign(fold=fold))
            print(f'Fold {fold}, {participant}: {len(df)} windows -> {out_dir}/{participant}_hr.png')
        fold_df = pd.concat(fold_df, ignore_index=True)
        metrics(fold_df).round(3).to_csv(f'{OUT_ROOT}/fold_{fold}/hr/hr_metrics.csv', index=False)
        all_df.append(fold_df)

    df = pd.concat(all_df, ignore_index=True)
    m = metrics(df)
    m.round(3).to_csv(f'{OUT_ROOT}/hr_metrics.csv', index=False)
    print(f'\nAll folds: {df.participant.nunique()} participants, {len(df)} windows\n')
    print(m.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
