# Preprocessed egoPPG-DB

Output of `preprocessing/preprocessing_egoppg.py` (config: `configs/preprocessing/config_preprocessing_egoppg.yml`)
run on the raw egoPPG-DB dataset (`/projects/aria_rppg/EgoPPG_Dataset`). This is the data PulseFormer is trained on.

The data (~33 GB) is excluded from git via `.gitignore`. Only this README is versioned.

```
preprocessed_egoppg/
└── CL128_W128_H48_LabelRaw_VideoTypeRaw/
    └── CL128_Down1_W128_H48_LabelDiffStandardized_VideoTypeDiffStandardized/
```

The folder name encodes the settings: chunks of 128 frames (~4.3 s at 30 fps), frames resized to 128×48,
no downsampling, and video and label in `DiffStandardized` format.

## Files per participant

25 participants (`001`–`025`). Each recording is split into chunks of 128 frames, and every chunk is saved as one
`.npy` file per signal. `<k>` is the chunk index in temporal order.

| File | Shape | Content |
|---|---|---|
| `<id>_input_et<k>.npy` | (1, 128, 48, 128) float32 | Eye-tracking video: frame difference, standardized |
| `<id>_input_imu_right<k>.npy` | (128,) | Right IMU: difference + standardization per axis, then magnitude |
| `<id>_label_ppg_nose<k>.npy` | (128,) | Label: cleaned nose PPG, difference + standardization |
| `<id>_label_classhr<k>.npy` | (128,) int | HR of the chunk in bpm, clipped to 40–180 (same value repeated). Not used by PulseFormer |

- Each participant has 414–499 chunks (~11,150 in total, ~13.2 h).
- Differencing and standardization are done per participant and per task, before chunking.
- All tasks are included. The selection of tasks and exclusions (`TASKS_TO_USE`, `exclusion_list`) is applied when
  loading the data (`ml/ml_helper.py`), not here.
