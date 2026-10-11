"""HR metrics vs BVP correlation, overall, per fold and for all folds.

Per fold, joins the overall row of hr/hr_metrics.csv (PulseFormer vs PPG nose: MAE, RMSE, MAPE, Pearson of the HR
per window) with the overall row of ppg/bvp_pearson_summary.csv (Pearson of the BVP waves, all samples pooled), and
adds the mean and std over the 5 folds.

Usage (from egoPPG-tricam/, after hr_ecg.py and bvp_pearson.py):
    python -m EVAL_TRAINING.comparison
"""
import pandas as pd

from EVAL_TRAINING.ppg_windows import OUT_ROOT

def overall_row(hr_path, bvp_path):
    hr = pd.read_csv(hr_path)
    hr = hr[hr.group == 'overall'].iloc[0]
    bvp = pd.read_csv(bvp_path, index_col='group').loc['overall']
    return {'n_windows': int(hr.n_windows), 'HR_MAE': hr.MAE, 'HR_RMSE': hr.RMSE, 'HR_MAPE': hr.MAPE,
            'HR_Pearson': hr.Pearson, 'BVP_Pearson': bvp.pearson_r}


def main():
    rows = {f'fold_{k}': overall_row(f'{OUT_ROOT}/fold_{k}/hr/hr_metrics.csv', f'{OUT_ROOT}/fold_{k}/ppg/bvp_pearson_summary.csv')
            for k in range(5)}
    df = pd.DataFrame(rows).T
    df.loc['mean_folds'] = df.mean()
    df.loc['std_folds'] = df.iloc[:5].std(ddof=0)
    df.loc[['mean_folds', 'std_folds'], 'n_windows'] = float('nan')
    df.round(3).to_csv(f'{OUT_ROOT}/comparison.csv', index_label='group')
    print(df.round(2).to_string())


if __name__ == '__main__':
    main()
