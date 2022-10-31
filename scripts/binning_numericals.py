'''
This script is used to modify mixed type datasets (numerical + categorical) so that the values in the numerical columns
are split in bins, so that imputation is carried out over the bins, rather than on the values themselves.
'''

import pandas as pd
import argparse
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_missing', type=str, required=True, action='store')
    parser.add_argument('--df_orig', type=str, required=True, action='store')
    parser.add_argument('--quantiles', type=int, default=[10], nargs='*', action='store')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--cut_columns', type=str, nargs='*', action='store')
    group.add_argument('--cut_numeric', action='store_true')

    args = parser.parse_args()
    return args


def return_cut_datasets(df_missing, df_orig, cut_cols, quantiles):
    bins = {col: None for col in cut_cols}
    for idx, col in enumerate(cut_cols):
        df_missing[col], bins[col] = pd.qcut(df_missing[col], quantiles[idx], retbins=True, duplicates='drop')
        df_orig[col] = pd.cut(df_orig[col], bins[col])
    return df_missing, df_orig


if __name__ == '__main__':
    args = parse_args()
    df_missing = pd.read_csv(args.df_missing)
    df_orig = pd.read_csv(args.df_orig)
    quantiles = args.quantiles

    if args.cut_numeric:
        cut_columns = df_missing.select_dtypes(include='number').columns
    else:
        cut_columns = args.cut_columns

    if len(quantiles) > 1:
        assert len(quantiles) == len(cut_columns)
    else:
        quantiles = quantiles*len(cut_columns)

    df_missing, df_orig = return_cut_datasets(df_missing, df_orig, cut_cols=cut_columns, quantiles=quantiles)

    m = {
        args.df_missing: df_missing,
        args.df_orig: df_orig
    }

    for path, df in m.items():
        dirpath, basename = osp.split(path)
        base, ext = osp.splitext(basename)
        case = base.split('_', maxsplit=1)
        if len(case) == 1:
            rest = ''
        else:
            rest = '_' + '_'.join(case[1:])
        case = case[0]
        fname = f'{dirpath}/{case}_binned{rest}{ext}'
        # fname = dirpath + '/'  + case + '_binned' +  + ext
        df.to_csv(fname, index=False)