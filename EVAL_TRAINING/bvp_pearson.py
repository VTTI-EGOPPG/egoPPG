"""Pearson correlation between the sensor and the estimated PPG waves, pooled over all samples.

Uses the predictions saved by EVAL_TRAINING/ppg_windows.py. Every non-overlapping 60 s window of the paper's evaluation
is reconstructed (cumsum, detrend, inversion, 0.7-2.8 Hz band-pass, z-score per window), so that both waves have the
same scale everywhere. Each sample keeps the activity of its own frame, so mixed windows contribute to each of their
activities. Pearson r is then computed on all samples pooled:
    - overall: all samples of all test participants of the fold(s)
    - per activity: all samples of that activity
    - per participant and activity
for every fold (results/fold_<k>/bvp_pearson_*.csv) and for all folds together (results/bvp_pearson_*.csv).

Usage (from egoPPG-tricam/, after ppg_windows.py has saved the predictions):
    python -m EVAL_TRAINING.bvp_pearson
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from EVAL_TRAINING.ppg_windows import CFG_PATH, OUT_ROOT, WINDOW, reconstruct, task_segments
from ml.config import get_config

ACTIVITY_ORDER = ['video', 'office', 'walking', 'kitchen', 'dancing', 'bike']


def participant_samples(config, participant, pred, label):
    """Reconstructed sensor and model waves of all 60 s windows, with the activity of every sample."""
    segments = task_segments(config, participant)
    tasks = np.concatenate([[name] * length for name, _, length in segments])
    sensor, model, activity = [], [], []
    for i in range(0, len(pred) - WINDOW + 1, WINDOW):
        sensor.append(reconstruct(label[i:i + WINDOW]))
        model.append(reconstruct(pred[i:i + WINDOW]))
        activity.append(tasks[i:i + WINDOW])
    return pd.DataFrame({'participant': participant, 'activity': np.concatenate(activity),
                         'sensor': np.concatenate(sensor), 'model': np.concatenate(model)})


def pearson(df):
    return pd.Series({'pearson_r': np.corrcoef(df.sensor, df.model)[0, 1], 'n_samples': len(df),
                      'minutes': len(df) / WINDOW})


def summarize(df, out_prefix):
    """Write <out_prefix>_summary.csv (overall + per activity) and <out_prefix>_per_participant.csv."""
    activities = [a for a in ACTIVITY_ORDER if a in set(df.activity)]
    summary = pd.concat([pearson(df).to_frame('overall').T,
                         df.groupby('activity')[['sensor', 'model']].apply(pearson).reindex(activities)])
    per_participant = (df.groupby(['fold', 'participant', 'activity'])[['sensor', 'model']].apply(pearson)
                       .reset_index())
    summary.round(3).to_csv(f'{out_prefix}_summary.csv', index_label='group')
    per_participant.round(3).to_csv(f'{out_prefix}_per_participant.csv', index=False)
    return summary, per_participant


def main():
    config = get_config(argparse.Namespace(cfg_path=CFG_PATH))

    samples = []
    for fold in range(5):
        fold_samples = []
        for f in sorted(glob.glob(f'{OUT_ROOT}/fold_{fold}/predictions/*.npy')):
            participant = os.path.basename(f).split('.')[0]
            pred, label = np.load(f)
            fold_samples.append(participant_samples(config, participant, pred, label).assign(fold=fold))
        if not fold_samples:
            print(f'Fold {fold}: no predictions found, skipped (run EVAL_TRAINING/ppg_windows.py first)')
            continue
        fold_df = pd.concat(fold_samples, ignore_index=True)
        summary, _ = summarize(fold_df, f'{OUT_ROOT}/fold_{fold}/bvp_pearson')
        print(f"Fold {fold}: overall r = {summary.loc['overall', 'pearson_r']:.3f}")
        samples.append(fold_df)
    df = pd.concat(samples, ignore_index=True)

    summary, per_participant = summarize(df, f'{OUT_ROOT}/bvp_pearson')
    print(f'\nAll folds: {df.participant.nunique()} participants, {len(df) / WINDOW:.0f} minutes\n')
    print(summary.round(3).to_string())
    print('\nPer participant and activity (pearson_r):')
    print(per_participant.pivot(index='participant', columns='activity', values='pearson_r')
          .reindex(columns=[a for a in ACTIVITY_ORDER if a in set(df.activity)]).round(2).to_string())


if __name__ == '__main__':
    main()
