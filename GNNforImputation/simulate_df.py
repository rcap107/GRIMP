import pandas as pd
import numpy as np
import string
import random
import os.path as osp
import os

## Case 0: all random
def case0(num_A, num_B):
    values_A = [f'A_{_}' for _ in range(num_A)]
    values_B = [f'B_{_}' for _ in range(num_B)]

    return values_A, values_B, None


## Case 1: A => B AND B => A
def case1(num_A, num_B):
    values_A = [f'A_{_}' for _ in range(num_A)]
    values_B = [[f'B_{_}'] for _ in range(num_B)]
    dep_A = dict(zip(values_A, values_B))
    print(dep_A)

    return values_A, values_B, dep_A


## Case 2: A => B
def case2(num_A, num_B):
    values_A = [f'A_{_}' for _ in range(num_A)]
    values_B = [f'B_{_}' for _ in range(num_B)]
    cand_B = [_ for _ in values_B]

    num_cand_A = {val: random.randint(1, 5) for val in values_A}
    dep_A={val: [ ] for val in values_A}

    for idx, val in enumerate(values_A):
        for _ in range(num_cand_A[val]):
            dep_A[val].append(cand_B.pop(random.randint(0, len(cand_B)-1)))

    print(dep_A)
    return values_A, values_B, dep_A

if __name__ == '__main__':
    num_rows = 3000
    num_dist_values_extra_cols = [5, -1, 15]
    num_A = 5
    num_B = 75
    extra_cols = [_ for _ in string.ascii_uppercase[2:2+len(num_dist_values_extra_cols)]]
    columns = ['A', 'B'] + extra_cols
    cases = [case0, case1, case2]

    sim_tag = 3

    odir = f'data/sim{num_rows}/'
    os.makedirs(odir, exist_ok=True)

    for idx, ca in enumerate(cases):
        fname = f'simulation{sim_tag }'
        values_A, values_B, dep_A = ca(num_A, num_B)
        rows=[]
        for r in range(num_rows):
            row = []
            a = random.choice(values_A)
            row.append(a)
            if dep_A is not None:
                b = random.choice(dep_A[a])
            else:
                b = random.choice(values_B)
            row.append(b)
            for _c, col in enumerate(extra_cols):
                if num_dist_values_extra_cols[_c] == -1:
                    s = f'{col}_{r}'
                else:
                    s = f'{col}_{random.randint(0,num_dist_values_extra_cols[_c]-1)}'
                row.append(s)
            rows.append(row)

        df = pd.DataFrame(rows, columns=columns)
        uniques = df.nunique()
        combs = '_'.join([f'{v}' for v in uniques])
        opath = osp.join(odir, f'simulation{sim_tag}_case{idx}_{combs}.csv')

        df.to_csv(f'data/sim3000/simulation3_case{idx}_{combs}.csv', index=False)
        # df.sample(n=1000).to_csv(f'data/sim3000/simulation3_1k_case{idx}_{combs}.csv', index=False)
        # df.sample(n=300).to_csv(f'data/sim3000/simulation3_3h_case{idx}_{combs}.csv', index=False)
