import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

result_dict = pickle.load(open('results/run_363.pkl', 'rb'))


checkpoints_mt = result_dict['checkpoints']['checkpoints_mt']
n_heads = result_dict['checkpoints']['init_params']['multitask_model']['input_tuple_length']


evolution = {idx: {n: {'q': None, 'k': None} for n in range(n_heads)
              } for idx in range(len(checkpoints_mt)) }

for idx, cp in enumerate(checkpoints_mt):
    case = {}
    for param in cp:
        if param.startswith('heads.'):
            split = param.split('.')
            if len(split) == 3:
                _, head_n, mat = split
                evolution[idx][int(head_n)][mat]=cp[param]


evo_q = {epoch: {} for epoch in range(len(evolution))}
evo_k = {epoch: {} for epoch in range(len(evolution))}
for epoch in evo_q:
    epo = evolution[epoch]
    it = 0
    # while epo[it]['q'] is None:
    #     it+=1
    epo_q = epo[it]['q']
    evo_q[epoch] = epo_q

    epo_k = {h: e['k'] for h, e in epo.items()}
    evo_k[epoch] = epo_k

fig, axs = plt.subplots(len(evolution), 1, figsize=(10,12))
for e, q in evo_q.items():
    if q is None:
        continue
    ax=axs[e]
    pos = ax.imshow(q, vmin=-0.5, vmax=0.5)
    fig.colorbar(pos, ax=ax)
fig.tight_layout()
plt.show()

fig, axs = plt.subplots(len(evolution), n_heads, figsize=(24,20))
for e, k_vals  in evo_k.items():
    for head, vals in k_vals.items():
        if vals is None:
            vals = np.zeros((n_heads, n_heads))
        ax=axs[e, head]
        pos = ax.imshow(vals, vmin=0, vmax=1)
        # fig.colorbar(pos, ax=ax)
fig.tight_layout()
plt.show()
