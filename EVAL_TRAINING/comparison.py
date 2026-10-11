"""HR metrics vs BVP correlation, overall, per fold and for all folds.

Joins the overall row of PulseFormer vs PPG nose in hr_metrics.csv (MAE, RMSE, MAPE, Pearson of the HR per window)
with the overall row of bvp_pearson_summary.csv (Pearson of the BVP waves, all samples pooled).

Usage (from egoPPG-tricam/, after hr_ecg.py and bvp_pearson.py):
    python -m EVAL_TRAINING.comparison
"""
import pandas as pd

from EVAL_TRAINING.ppg_windows import OUT_ROOT

HR_COMPARISON = 'PulseFormer vs PPG nose (sensor)'


def overall_row(hr_path, bvp_path):
    hr = pd.read_csv(hr_path)
    hr = hr[(hr.comparison == HR_COMPARISON) & (hr.group == 'overall')].iloc[0]
    bvp = pd.read_csv(bvp_path, index_col='group').loc['overall']
    return {'n_windows': int(hr.n_windows), 'HR_MAE': hr.MAE, 'HR_RMSE': hr.RMSE, 'HR_MAPE': hr.MAPE,
            'HR_Pearson': hr.Pearson, 'BVP_Pearson': bvp.pearson_r}


def main():
    rows = {f'fold_{k}': overall_row(f'{OUT_ROOT}/fold_{k}/hr/hr_metrics.csv', f'{OUT_ROOT}/fold_{k}/bvp_pearson_summary.csv')
            for k in range(5)}
    df = pd.DataFrame(rows).T
    df.loc['mean_folds'] = df.mean()
    df.loc['std_folds'] = df.iloc[:5].std(ddof=0)
    df.loc['all_folds'] = overall_row(f'{OUT_ROOT}/hr_metrics.csv', f'{OUT_ROOT}/bvp_pearson_summary.csv')
    df.loc[['mean_folds', 'std_folds'], 'n_windows'] = float('nan')
    df.round(3).to_csv(f'{OUT_ROOT}/comparison.csv', index_label='group')
    print(df.round(2).to_string())


if __name__ == '__main__':
    main()
