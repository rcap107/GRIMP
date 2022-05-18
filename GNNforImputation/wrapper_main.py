'''
This script is used for performing grid search given the dataset


Author: Riccardo Cappuzzo
'''
from itertools import product
import os.path as osp
import os
from copy import deepcopy
from main_multilabel import wrapper
from argparse import Namespace


def prepare_parameters(par_dict, default_pars):
    all_pars = []
    for comb in product(*par_dict.values()):
        updated_pars = deepcopy(default_pars)
        this_dict = dict(zip(par_dict.keys(), comb))
        for key, value in this_dict.items():
            updated_pars[key] = value
        all_pars.append(updated_pars)
    return all_pars



# # Dataset names, only the datasets in this list will be used for the experiments.
# datasets = ['adultsample10',
#             'australian',
#             'contraceptive',
#             'credit',
#             'flare',
#             'mammogram',
#             'thoracic',
#             'tictactoe'
# ]
datasets = ['tictactoe',
            'imdb',
            'tax5000trimmed'
]

CLEAN_DIR = 'data/clean'
DIRTY_DIR = 'data/dirty'
PRETRAINED_DIR = 'data/pretrained-emb'
all_cases = []

for f in os.listdir('data/clean'):
    case = {
        'ground_truth': None,
        'dirty_dataset': []
    }
    dsname, ext = osp.splitext(f)
    if dsname not in datasets:
        continue
    print(dsname)

    case['ground_truth'] = osp.join(CLEAN_DIR,f)
    if not osp.exists(case['ground_truth']):
        print(f'File {f} not found.')
        continue
    for dirty_f in os.listdir('data/dirty'):
        dirty_dataset_case, ext = osp.splitext(dirty_f)
        dirty_name, tgt_cols, error_frac = dirty_dataset_case.split('_')
        if dirty_name == dsname:
            # for pretrained_case in [['ft', 'embdi_f4'] ]:
            for pretrained_case in [['ft'], ['embdi_f4'], ['ft', 'embdi_f4'] ]:
                emb_full_paths = []
                for sub_case in pretrained_case:
                    pretrained_path = f'{dirty_name}_{error_frac}_{sub_case}.emb'
                    pretrained_full_path = osp.join(PRETRAINED_DIR, pretrained_path)
                    if osp.exists(pretrained_full_path):
                        emb_full_paths.append(pretrained_full_path)

                case['dirty_dataset']= osp.join(DIRTY_DIR, dirty_f)
                case['text_embs'] = emb_full_paths
                case['imputed_df_tag'] = ''.join(pretrained_case)
                all_cases.append({k:v for k,v in case.items()})

# Sorting the list in alphabetic order.
all_cases.sort(key=lambda x: x['dirty_dataset'])

if len(all_cases) == 0:
    raise ValueError(f'No cases have been generated. ')

arg_keys = [
    'ground_truth',
    'dirty_dataset',
    'graph_layers',
    'gnn_feats',
    'jumping_knowledge',
    'h_feats',
    'predictor_layers',
    'head_layers',
    'training_subset',
    'target_columns',
    'training_columns',
    'ignore_columns',
    'cat_columns',
    'ignore_num_cols',
    'flag_col',
    'flag_rid',
    'loss',
    'loss_alpha',
    'loss_gamma',
    'module_aggr',
    'heteroconv_aggr',
    'epochs',
    'grace',
    'dropout_gnn',
    'dropout_clf',
    'learning_rate',
    'weight_decay',
    'comb_size',
    'max_comb_num',
    'shared_model',
    'head_model',
    'node_features',
    'node_dim',
    'random_init',
    'fd_path',
    'fd_strategy',
    'no_relu',
    'no_sm'
    'save_imputed_df',
    'imputed_df_tag',
    'skip_gnn',
    'text_embs'
]

default_parameters = {
                 'architecture': 'multitask',
                 'graph_layers': 2,
                 'gnn_feats': 16,
                 'jumping_knowledge': False,
                 'h_feats': 32,
                 'predictor_layers': 2,
                 'head_layers': 1,
                 'training_subset': 'target',
                 'target_columns': None,
                 'training_columns': None,
                 'ignore_columns': None,
                 'cat_columns': [],
                 'ignore_num_cols': False,
                 'flag_col': False,
                 'flag_rid': False,
                 'loss': 'xe',
                 'loss_alpha': 0.5,
                 'loss_gamma': 2,
                 'module_aggr': 'gcn',
                 'heteroconv_aggr': 'sum',
                 'epochs': 300,
                 'grace': 150,
                 'dropout_gnn': 0,
                 'dropout_clf': 0,
                 'learning_rate': 0.001,
                 'th_stop': 1e-5,
                 'force_training': False,
                 'weight_decay': 1e-4,
                 'comb_size': 1,
                 'max_comb_num': 10,
                 'shared_model': 'linear',
                 'head_model': 'attention',
                 'node_features': None,
                 'max_components': 300,
                 'random_init': False,
                 'fd_path': None,
                 'fd_strategy': 'attention',
                 'no_relu': False,
                 'no_sm': False,
                 'save_imputed_df': True,
                 'training_sample': 1,
                 'batchnorm': False,
                 'k_strat': 'weak',
                 'imputed_df_tag': '',
                 'skip_gnn': False,
                'seed':None,
                'load_model_file_path': '',
                'save_model_file_path': '',
                'text_embs': '',
                'np_mat': None,

    }

# Attention linear, attention head
parameters_default= {
    'graph_layers': [2],
    'gnn_feats': [64],
    'h_feats': [32],
    'shared_model': ['linear'],
    'head_model': ['attention'],
    'k_strat': ['weak'],
    'skip_gnn': [False],
    'random_init': [False]
}

num_cases = 0
for case in all_cases:
    default_parameters.update(case)
    all_pars = prepare_parameters(parameters_default, default_parameters)
    for run in range(3):
        for pars in all_pars:
            print(case)
            try:
                wrapper(Namespace(**pars))
            except Exception as e:
                print(e)
                print(f'Run {num_cases} failed')
            num_cases+=1
            os.system('cp /content/GNNforImputation/results/results.csv   /content/drive/MyDrive/Colab\ Notebooks/results.csv')

print(num_cases)