import numpy as np
import matplotlib.pyplot as plt


n_col = 5
tgt_col = 1
fd_col = [1, 2]

# Diagonal, all columns are present with the same weight.
def f_mat1(n_col):
    return np.eye(n_col)


mat1 = f_mat1(n_col)

# Only tgt_col is present.
def f_mat2(n_col, tgt_col):
    mat = np.zeros((n_col, n_col))
    mat[tgt_col, tgt_col] = 1
    return mat


mat2 = f_mat2(n_col, tgt_col)

# All columns present, tgt_col is boosted.
def f_mat3(n_col, tgt_col):
    mat = np.eye(n_col)
    mat *= 0.1
    mat[tgt_col, tgt_col] = 1
    return mat


mat3 = f_mat3(n_col, tgt_col)

# All columns, FDs boosted.
def f_mat4(n_col, tgt_col, fd_col):
    mat = np.eye(n_col)
    mat *= 0.1
    if tgt_col in fd_col:
        mat[fd_col, fd_col] = 0.5
    mat[tgt_col, tgt_col] = 1
    return mat


mat4 = f_mat4(n_col, tgt_col, fd_col)

# Only FDs
def f_mat5(n_col, tgt_col, fd_col):
    mat = np.zeros((n_col, n_col))
    if tgt_col in fd_col:
        mat[fd_col, fd_col] = 0.5
    mat[tgt_col, tgt_col] = 1
    return mat


mat5 = f_mat5(n_col, tgt_col, fd_col)


def f_wrapper(n_col, tgt_col, fd_col, case):
    if case == 1:
        return f_mat1(n_col)
    elif case == 2:
        return f_mat2(n_col, tgt_col)
    elif case == 3:
        return f_mat3(n_col, tgt_col)
    elif case == 4:
        return f_mat4(n_col, tgt_col, fd_col)
    elif case == 5:
        return f_mat5(n_col, tgt_col, fd_col)


mats = [mat1, mat2, mat3, mat4, mat5]
titles = [
    "Diagonal",
    "Target column",
    "Weak diagonal",
    "Weak diagonal + FD",
    "Target column + FD",
]

fig, axs = plt.subplots(1, 5, figsize=(10, 3))
for idx, ax in enumerate(axs):
    pos = ax.imshow(mats[idx], vmin=0, vmax=1)
    ax.set_title(f"{titles[idx]}")
    ax.set_xticks(range(0, 5), labels=range(1, 6))
    ax.set_yticks(range(0, 5), labels=range(1, 6))
# plt.colorbar(pos, ax=axs, location='bottom')
fig.tight_layout()
plt.show()
fig.savefig("k_variants.png")
#
# # Plotting each strategy separately.
# for idx, mat in enumerate(mats):
#     fig = plt.figure( figsize=(3, 3))
#     ax = plt.imshow(mats[idx], vmin=0, vmax=1)
#     plt.title(f'{titles[idx]}')
#     plt.colorbar(pos, ax=axs, location='bottom')
#     # fig.tight_layout()
#     plt.show()


# Plotting the full set of columns
# fig, axs = plt.subplots(5, n_col, figsize=(10, 6))
# for idx1, strat in enumerate(titles):
#     for idx2 in range(n_col):
#         ax = axs[idx1, idx2]
#         mat = f_wrapper(n_col, idx2, fd_col, case=idx1+1)
#         pos = ax.imshow(mat, vmin=0, vmax=1)
#         ax.xaxis.set_ticklabels([])
#         ax.xaxis.set_ticklabels([])
# axs[idx1,:].set_title(f'{strat}')
# plt.colorbar(pos, ax=axs, location='bottom')
# fig.tight_layout()
plt.show()
