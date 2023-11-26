import pandas as pd
import argparse
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", "-i", type=str, required=True)
    parser.add_argument("--fd_path", type=str, required=True)

    args = parser.parse_args()

    return args


def read_csv(input_path):
    df = pd.read_csv(input_path)
    return df


def read_functional_dependencies(fd_filename, df):
    # As implemented in MissForestFD
    with open(fd_filename) as f:
        lines = f.readlines()
    fd_dict = {}
    df_columns = list(df.columns)

    for line in lines:
        # FD is of the form a,b,c->d
        lhs, rhs = line.strip().split("->")
        lhs = lhs.split(",")
        # Find the indices of FDs from the list of attributes
        lhs_attr = tuple(lhs)
        if rhs not in fd_dict:
            fd_dict[rhs] = () + (lhs_attr,)
        else:
            # Insert as tuple of  tuples
            # E.g. fd_dict[1] = ( (2,3), (3,4,5) )
            fd_dict[rhs] = fd_dict[rhs] + (lhs_attr,)
    return fd_dict


def generate_fd_mapping(df, fd_dict):
    full_mapping = dict()

    for rhs in fd_dict:
        lhs_c = fd_dict[rhs]
        full_mapping[rhs] = {lhs: None for lhs in lhs_c}
        for lhs in lhs_c:
            col_subset = list(lhs + (rhs,))
            df_subset = df[col_subset].dropna(axis=0)
            groups = df_subset.groupby(list(lhs))
            mapping = dict()
            for name, group in groups:
                uq = group[rhs].unique()[0]
                if len(lhs) == 1:
                    mapping[(name,)] = uq
                else:
                    mapping[name] = uq
            full_mapping[rhs][lhs] = mapping
    return full_mapping


def fix_errors(df: pd.DataFrame, fd: dict, mapping):
    df_orig = df.copy()
    for idx, row in df.iterrows():
        for col in df.columns:
            if pd.isna(row[col]):
                if col in mapping:
                    for lhs in mapping[col]:
                        vals = tuple(row[list(lhs)].values.tolist())
                        if vals in mapping[col][lhs]:
                            df.loc[idx, col] = mapping[col][lhs][vals]
                            continue
    return df


def save_df(input_dataset, fixed_dataset):
    base, ext = osp.splitext(input_dataset)
    out_fname = base + "_fixed" + ext
    fixed_dataset.to_csv(out_fname, index=False)


if __name__ == "__main__":
    args = parse_args()

    df = read_csv(args.input_dataset)
    fd_dict = read_functional_dependencies(args.fd_path, df)

    mapping = generate_fd_mapping(df, fd_dict)

    df_fixed = fix_errors(df, fd_dict, mapping)
    save_df(args.input_dataset, df_fixed)
