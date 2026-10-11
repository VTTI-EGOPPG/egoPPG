"""Sensor vs predicted PPG for the test participants of one egoPPG-DB fold.

For every test participant of the fold, the best model of that fold (models/fold_<k>/*.pth) predicts the PPG of all
kept chunks. The concatenated signal is split into the paper's evaluation windows (non-overlapping 60 s windows, as in
evaluation/metrics_ppg.py). For every activity, the first window that contains only that activity is reconstructed as in the paper (cumsum, detrend, inversion, 0.7-2.8 Hz band-pass) and plotted:
    - nose PPG sensor (ground truth) vs PulseFormer estimate, with the detected peaks
    - HR of each wave (peak detection, same as evaluation/post_process.py)
    - Pearson correlation between the two waves

Step 1 needs a GPU and is skipped if the predictions already exist. Step 2 runs on CPU.

Usage (from egoPPG-tricam/):
    python -m EVAL_TRAINING.ppg_windows --fold 1
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from scipy.signal import butter, detrend, filtfilt, find_peaks
from torch.utils.data import DataLoader

from evaluation.post_process import calculate_metric_per_video_ppg
from ml import ml_helper
from ml.config import get_config
from ml.data_loader.DatasetLoader import DatasetLoader
from ml.models.PulseFormer import PulseFormer
from utils import get_adjusted_task_times, get_participants_lists

CFG_PATH = 'configs/ml/egoppg_egoppg_PulseFormer.yaml'
CFG_PRE_PATH = 'configs/preprocessing/config_preprocessing_egoppg.yml'
MODELS_DIR = 'models'
OUT_ROOT = 'EVAL_TRAINING/results'
SEED = 0
FS = 30
WINDOW = 60 * FS
TASK_COLORS = {'video': 'tab:blue', 'office': 'tab:orange', 'walking': 'tab:brown', 'kitchen': 'tab:green',
               'dancing': 'tab:red', 'bike': 'tab:purple'}


def predict_fold(config, fold, out_dir):
    """Run the best model of the fold on its test participants and save <id>.npy = [prediction, label]."""
    participants, participants_test = get_participants_lists(config)
    cv_split_files = ml_helper.get_cv_split_files(config, participants, participants_test, SEED)
    split = f'kfold{config.K_FOLD_SPLITS}_{fold}'
    dataset = DatasetLoader('test', cv_split_files['test'][split], config.INPUT_SIGNALS, config.LABEL_SIGNALS,
                            config.TEST.DATA)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)

    pth = glob.glob(f'{MODELS_DIR}/fold_{fold}/*.pth')
    assert len(pth) == 1, pth
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PulseFormer(frames=config.TEST.DATA.PREPROCESS.CHUNK_LENGTH).to(device)
    model.load_state_dict(torch.load(pth[0], map_location=device, weights_only=True))
    model.eval()
    print(f'Fold {fold}: {pth[0]} on {device}, {len(dataset)} test chunks')

    preds, labels = {}, {}
    with torch.no_grad():
        for inputs, label, participants_b, chunk_ids in loader:
            pred = model(inputs[0].to(device), inputs[1].to(device)).cpu().numpy()
            for j in range(len(participants_b)):
                preds.setdefault(participants_b[j], {})[int(chunk_ids[j])] = pred[j].reshape(-1)
                labels.setdefault(participants_b[j], {})[int(chunk_ids[j])] = label[0][j].numpy()

    # Concatenate chunks in temporal order (same as _reform_data_from_dict in evaluation/metrics_ppg.py)
    for p in preds:
        pred = np.concatenate([preds[p][k] for k in sorted(preds[p])])
        label = np.concatenate([labels[p][k] for k in sorted(labels[p])])
        np.save(f'{out_dir}/{p}.npy', np.asarray([pred, label]))
        print(f'Saved {p}: {pred.shape[0]} frames')


def task_segments(config, participant):
    """(activity, start, length) of every kept task segment in the concatenated prediction, in recording order.
    Same chunk counting and task selection as utils.get_task_chunk_list, but walking_1/2/3 stay separate."""
    with open(CFG_PRE_PATH) as f:
        cfg_pre = yaml.load(f, Loader=yaml.FullLoader)
    chunk = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
    segments, start = [], 0
    for task, (t0, t1) in get_adjusted_task_times(cfg_pre['task_times'][participant]).items():
        name = 'walking' if task.startswith('walking') else task
        n_chunks = ((t1 - t0) // config.TRAIN.DATA.PREPROCESS.DOWNSAMPLE + 1) // chunk
        if name in config.TASKS_TO_USE and participant not in cfg_pre['exclusion_list'][name]:
            segments.append((name, start, n_chunks * chunk))
            start += n_chunks * chunk
    return segments


def reconstruct(x):
    """DiffStandardized -> PPG wave, same steps as calculate_metric_per_video_ppg, z-scored for plotting."""
    x = detrend(np.cumsum(x))
    x = np.max(x) - x
    b, a = butter(4, [0.7 / FS * 2, 2.8 / FS * 2], btype='bandpass')
    x = filtfilt(b, a, np.double(x))
    return (x - x.mean()) / x.std()


def activity_windows(config, participant, pred, label):
    """The paper's evaluation windows (evaluation/metrics_ppg.py): non-overlapping 60 s windows over the concatenated
    signal, incomplete last window dropped. Windows that span more than one activity are skipped; for every activity,
    the first window that contains only that activity is kept (activities without such a window are not shown)."""
    segments = task_segments(config, participant)
    assert sum(s[2] for s in segments) == len(pred), (participant, sum(s[2] for s in segments), len(pred))
    tasks = np.concatenate([[name] * length for name, _, length in segments])
    rows, seen = [], set()
    for i_window, i in enumerate(range(0, len(pred) - WINDOW + 1, WINDOW)):
        activities = set(tasks[i:i + WINDOW])
        if len(activities) > 1:
            continue
        activity = activities.pop()
        if activity in seen:
            continue
        seen.add(activity)
        p_win, l_win = pred[i:i + WINDOW], label[i:i + WINDOW]
        hr_gt, hr_pred, _, _, _ = calculate_metric_per_video_ppg(p_win, l_win, 'Peak_Detection', True, FS)
        p_wave, l_wave = reconstruct(p_win), reconstruct(l_win)
        rows.append({'participant': participant, 'window': i_window, 'activity': activity,
                     'hr_sensor': hr_gt, 'hr_model': hr_pred, 'pearson_r': np.corrcoef(l_wave, p_wave)[0, 1],
                     'wave_sensor': l_wave, 'wave_model': p_wave})
    return rows


def plot_participant(rows, fold, out_dir):
    participant = rows[0]['participant']
    fig, axes = plt.subplots(len(rows), 1, figsize=(20, 2.8 * len(rows)), squeeze=False)
    for ax, r in zip(axes[:, 0], rows):
        t = np.arange(len(r['wave_sensor'])) / FS
        for wave, color, name in [(r['wave_sensor'], 'black', 'Sensor (nose PPG)'),
                                  (r['wave_model'], 'tab:red', 'PulseFormer')]:
            ax.plot(t, wave, color=color, lw=1.2, label=name)
            peaks, _ = find_peaks(wave)
            ax.plot(t[peaks], wave[peaks], 'v', color=color, ms=5)
        ax.set_title(f"{r['activity']} (window {r['window']})  |  HR sensor = {r['hr_sensor']:.1f} bpm, HR model = {r['hr_model']:.1f} bpm, "
                     f"|error| = {abs(r['hr_model'] - r['hr_sensor']):.1f} bpm  |  Pearson r = {r['pearson_r']:.2f}",
                     fontsize=11, color=TASK_COLORS[r['activity']], loc='left')
        ax.set_xlim(0, t[-1])
        ax.set_ylabel('PPG (z-score)')
        ax.legend(fontsize=8, loc='upper right')
    axes[-1, 0].set_xlabel('time from window start (s)')
    fig.suptitle(f'Participant {participant} (fold {fold}): first single-activity 60 s window of each activity', fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/{participant}_ppg_windows.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    args = parser.parse_args()
    config = get_config(argparse.Namespace(cfg_path=CFG_PATH))

    pred_dir = f'{OUT_ROOT}/fold_{args.fold}/predictions'
    out_dir = f'{OUT_ROOT}/fold_{args.fold}/ppg'
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    if not glob.glob(f'{pred_dir}/*.npy'):
        predict_fold(config, args.fold, pred_dir)

    all_rows = []
    for f in sorted(glob.glob(f'{pred_dir}/*.npy')):
        participant = os.path.basename(f).split('.')[0]
        pred, label = np.load(f)
        rows = activity_windows(config, participant, pred, label)
        plot_participant(rows, args.fold, out_dir)
        all_rows += rows
        print(f'{participant}: {len(rows)} activities -> {out_dir}/{participant}_ppg_windows.png')

    df = pd.DataFrame(all_rows).drop(columns=['wave_sensor', 'wave_model'])
    print(df.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
