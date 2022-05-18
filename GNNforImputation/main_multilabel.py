'''

Author: XXX
'''

import argparse
import os.path as osp
import pickle
import random
import time

import numpy as np
import torch
from numpy import inf

import GRIMP.gnn_models as gnn_models
from GRIMP.multitask_predictor import MultiTaskPredictor
import GRIMP.testing_utils as testing_utils
from GRIMP.gnn_utils import convert_embs_to_text, read_external_features, save_model_to_file, \
    load_model_from_file, read_features_tensor, read_functional_dependencies
from GRIMP.logging import GrimpLogger as Logger
from GRIMP.multilabel_graph_dataset import ImputationTripartiteGraphMultilabelClassifier
import GRIMP.multilabel_training as mlt
from GRIMP.heterograph_dataset import HeterographDataset

logger = Logger()


def parse_args():
    parser = argparse.ArgumentParser()

    # Datasets
    parser.add_argument('--ground_truth', action='store', type=str, default=None,
                        help='Clean version of the dataset to measure the imputation accuracy. ')
    parser.add_argument('--dirty_dataset', action='store', type=str, required=True,
                        help='Dataset containing missing values to impute. Missing values should be left empty.')

    # Architecture parameters
    parser.add_argument('--architecture', action='store', default='multitask', choices=['multilabel', 'multitask'],
                        type=str)
    parser.add_argument('--graph_layers', action='store', default=2, type=int)
    parser.add_argument('--gnn_feats', default=16, action='store', type=int)
    parser.add_argument('-j', '--jumping_knowledge', action='store_true')

    parser.add_argument('--h_feats', default=32, action='store', type=int)
    parser.add_argument('--predictor_layers', action='store', default=2, type=int)
    parser.add_argument('--head_layers', action='store', default=1, type=int)
    parser.add_argument('--training_subset', action='store', default='target', choices=['all', 'missing', 'target'])
    parser.add_argument('--target_columns', action='store', default=None, nargs='*')
    parser.add_argument('--training_columns', action='store', default=None, nargs='*')
    parser.add_argument('--ignore_columns', action='store', default=None, nargs='*')
    parser.add_argument('--cat_columns', action='store', default=[], nargs='*',
                        help='Specify which numerical columns should be treated as categorical.')

    parser.add_argument('--no_relu', action='store_true',
                        help='Whether to add relus at the output of the Q/K linear layers.')
    parser.add_argument('--no_sm', action='store_true',
                        help='Whether to add SM at the output of the M pooling. ')


    parser.add_argument('--ignore_num_cols', action='store_true')
    parser.add_argument('--flag_col', action='store_true',
                        help='Use this flag to replace missing values in a column with that column\'s vector.')
    parser.add_argument('--flag_rid', action='store_true',
                        help='Use this flag to add the RID vector at the beginning of each training sample. ')

    parser.add_argument('--loss', action='store', choices=['xe', 'focal'], default='xe',
                        help='Which loss to use, either cross entropy (xe) or focal loss (focal).')
    parser.add_argument('--loss_alpha', action='store', default=0.5, type=float,
                        help='Value of alpha to be used in the focal loss.')
    parser.add_argument('--loss_gamma', action='store', default=2, type=float,
                        help='Value of gamma to be used in the focal loss.')
    parser.add_argument('--module_aggr', action='store', default='gcn')
    parser.add_argument('--heteroconv_aggr', action='store', default='sum')

    # Training parameters
    parser.add_argument('--epochs', default=1000, action='store', type=int)
    parser.add_argument('--dropout_gnn', action='store', default=0, type=float)
    parser.add_argument('--dropout_clf', action='store', default=0, type=float)
    parser.add_argument('--batchnorm', action='store_true', default=False,
                        help='Whether to add a BatchNorm1d layer to the Multitask Attribute section.')
    parser.add_argument('--learning_rate', action='store', default=0.001, type=float)
    parser.add_argument('--weight_decay', action='store', default=1e-4, type=float)
    parser.add_argument('--th_stop', action='store', default=1e-5, type=float)
    parser.add_argument('--grace', action='store', default=1000, type=int,
                        help='Minimum number of training epochs before early stopping can trigger.')
    parser.add_argument('--force_training', action='store_true',
                        help='Use this flag to ignore early stopping. ')
    parser.add_argument('--comb_size', action='store', type=int, default=1,
                        help='Needed to use additional training samples. '
                             'Maximum size of column combinations to be generated. Check README for more info.')
    parser.add_argument('--max_comb_num', action='store', type=int, default=10,
                        help='Maximum number of combinations to be used. ')
    parser.add_argument('--training_sample', action='store', type=int, default=1.0,
                        help='Fraction of training samples to be used. Reducing this value will likely worsen results.')
    parser.add_argument('--shared_model', action='store', choices=['linear', 'attention'], default='attention')
    parser.add_argument('--head_model', action='store', choices=['linear', 'attention'], default='attention')
    parser.add_argument('--k_strat', action='store', choices=['full', 'single', 'weak'], default='weak')

    # Additional parameters
    parser.add_argument('--text_embs', action='store', type=str, default=None, nargs='*',
                        help='Load embeddings from word2vec format.')
    parser.add_argument('--np_mat', action='store', type=str, default=None,
                        help='Load embeddings from numpy matrix.')
    parser.add_argument('--max_components', action='store', type=int, default=300,
                        help='Number of node features components.')

    parser.add_argument('--output_emb_file', action='store', type=str)
    parser.add_argument('--random_init', action='store_true', default=False)
    parser.add_argument('--fd_path', action='store', default=None,
                        help='Path to the formatted file that contains the FDs.')
    parser.add_argument('--fd_strategy', action='store', choices=['val2val', 'col2col', 'v2vc2c', 'attention'],
                        default='attention',
                        help='Strategy to use when adding FD edges to the graph. ')

    # Loading/saving model
    parser.add_argument('--dump_graph', action='store_true', default=None,
                        help='DEBUG ONLY: create the graph dataset structure, dump it with pickle and exit the program.')
    parser.add_argument('--load_graph', action='store', type=str, default=None)
    parser.add_argument('--save_model_file_path', action='store', default=None, type=str)
    parser.add_argument('--load_model_file_path', action='store', default=None, type=str)

    # Saving imputed dataset
    parser.add_argument('--save_imputed_df', action='store_true', default=False,
                        help='Whether to save on file the imputed dataset.')
    parser.add_argument('--imputed_df_tag', action='store', default='',
                        help='Tag that will be added to the name of the imputed df.')


    parser.add_argument('--seed', action='store', default=None, type=int,
                    help='Random seed for reproducibility.')

    parser.add_argument('--skip_gnn', action='store_true',
                        help='Parameter for testing the training with no GNN.')


    args = parser.parse_args()

    return args


def set_random_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_graph_dataset(load_graph_dataset_path):
    graph_dataset = pickle.load(open(load_graph_dataset_path, 'rb'))
    return graph_dataset


def create_graph_dataset(args: argparse.Namespace):
    start = time.time()
    logger.add_time('start_graph_creation')
    print("creating graph from scratch")

    original_data_file_name = args.ground_truth
    missing_value_file_name = args.dirty_dataset

    # Ensure that both input files exist.
    if args.ground_truth is not None:
        assert osp.exists(original_data_file_name)
    assert osp.exists(missing_value_file_name)

    random_init = args.random_init

    if args.text_embs:
        node_mapping, ext_features = read_external_features(missing_value_file_name, args.text_embs, args.max_components)
    elif args.np_mat:
        ext_features = read_features_tensor(args.np_mat)
        node_mapping = None
    else:
        ext_features = None
        node_mapping = None

    if args.fd_path:
        fds = read_functional_dependencies(args.fd_path, args.dirty_dataset)
    else:
        fds = None

    architecture = 'multitask'

    graph_dataset = HeterographDataset(original_data_file_name,
                                       missing_value_file_name,
                                       random_init,
                                       node_mapping, ext_features, architecture=architecture,
                                       training_subset=args.training_subset,
                                       training_columns=args.training_columns,
                                       ignore_columns=args.ignore_columns,
                                       ignore_num_flag=args.ignore_num_cols,
                                       convert_columns=args.cat_columns,
                                       target_columns=args.target_columns,
                                       fd_dict=fds, fd_strategy=args.fd_strategy,
                                       max_comb_size=args.comb_size,
                                       max_num_comb=args.max_comb_num,
                                       flag_rid=args.flag_rid, flag_col=args.flag_col,
                                       training_sample=args.training_sample,
                                       )
    logger.add_time('end_graph_creation')
    logger.add_duration('start_graph_creation', 'end_graph_creation', 'graph_duration')

    if args.save_model_file_path:
        pickle.dump(graph_dataset, open(args.save_model_file_path, 'wb'))
    print("creating graph took ", time.time() - start)

    return graph_dataset


def run_training(args: argparse.Namespace):
    '''
    Train the imputation model following the arguments passed in args.
    :param args: User arguments passed on the command line.
    :return:
    '''
    if args.seed is not None:
        set_random_seeds(args.seed)

    if args.load_model_file_path:
        graph_dataset = load_graph_dataset(args.load_model_file_path)
    else:
        graph_dataset = create_graph_dataset(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_dataset.graph = graph_dataset.graph.to(device)
    graph_stats = graph_dataset.get_statistics()
    if graph_stats['node_features'] == 'external':
        if args.text_embs is not None:
            graph_stats['node_features'] = args.text_embs
        elif args.np_mat is not None:
            graph_stats['node_features'] = args.np_mat
        else:
            raise ValueError
    if args.fd_path is not None:
        graph_stats['fd_path'] = args.fd_path
    else:
        graph_stats['fd_path'] = None
    logger.add_dict('statistics', graph_stats)
    logger.add_value('parameters', 'training_columns', logger.get_value('statistics', 'training_columns'))
    # graph_dataset.graph.ndata['features'].to(device)

    imputation_columns = graph_dataset.training_columns

    if not args.skip_gnn:
        gnn_params = { 'gnn_in_feats': graph_dataset.num_features,
                        'gnn_h_feats': args.gnn_feats,
                        'num_layers': args.graph_layers,
                        'build_columns': graph_dataset.all_columns,
                        'build_fds': graph_dataset.fd_list,
                        'heteroconv_aggr':args.heteroconv_aggr,
                        'module_aggr':args.module_aggr,
                        'dropout':args.dropout_gnn
        }
        gnn_model = gnn_models.HeteroGraphSAGE(**gnn_params).to(device)
    else:
        gnn_model = torch.nn.Linear(graph_dataset.num_features, args.gnn_feats).to(device)
        gnn_params = {'in_features': graph_dataset.num_features,
                      'out_features': args.gnn_feats}

    distinct_values = []
    for col in graph_dataset.all_columns:
        if col in imputation_columns:
            col_id = graph_dataset.col2idx[col]
            distinct_values.append(graph_dataset.head_dims[col_id])
        else:
            distinct_values.append(None)

    if args.jumping_knowledge:
        gnn_feats = args.gnn_feats * args.graph_layers
    else:
        gnn_feats = args.gnn_feats

    mt_params = {
        'shared_in_feats':gnn_feats,
        'shared_out_feats':gnn_feats,
        'shared_h_feats':args.h_feats,
        'shared_h_layers':args.predictor_layers,
        'head_h_layers':args.head_layers,
        'head_out_feats_list':distinct_values,
        'input_tuple_length':graph_dataset.input_tuple_length,
        'dropout':args.dropout_clf, 'batchnorm':args.batchnorm,
        'shared_model':args.shared_model, 'head_model':args.head_model,
        'graph_dataset':graph_dataset,
        'k_strat': args.k_strat,
        'no_relu': args.no_relu,
        'no_sm': args.no_sm,
        'device':device
    }

    prediction_model = MultiTaskPredictor(**mt_params).to(device)
    logger.add_value('parameters', 'predictor_structure', prediction_model.get_model_structure())
    prediction_model = prediction_model.to(device)
    if not args.skip_gnn:
        logger.add_value('parameters', 'gnn_structure', gnn_model.get_model_name())
    else:
        logger.add_value('parameters', 'gnn_structure', 'liner')
    init_parameters = {
        'gnn_model': gnn_params,
        'multitask_model': mt_params,
    }

    states = {
        'init_params': init_parameters,
        'checkpoints_gnn': [],
        'checkpoints_mt': []
    }

    print("Begin training")
    start = time.time()
    logger.add_time('start_training')
    logger.add_dict('curves', {
        'loss': [],
        'min': inf,
        'end': 0,
        'min_epoch': -1,
        'loss_valid': [],
        'min_valid': inf,
        'end_valid': 0,
        'min_epoch_valid': -1,
    })

    # logger.add_dict('checkpoints', states)

    best_state = mlt.train(graph_dataset, gnn_model, multitask_model=prediction_model, args=args,
                                     device=device, logger=logger)

    print("training took ", time.time() - start)
    logger.add_time('end_training')
    logger.add_duration('start_training', 'end_training', 'training_duration')

    # if args.output_emb_file:
    #     convert_embs_to_text(best_state, graph_dataset, args.emb_file)
    # if args.save_model_file_path:
    #     save_model_to_file(args.save_model_file_path, graph_dataset, gnn_model, prediction_model)


    return graph_dataset, best_state, init_parameters


def run_testing(args=None, graph_dataset=None, best_state=None, init_params=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Begin testing")
    start = time.time()
    logger.add_time('start_testing')

    df_imputed = testing_utils.generate_imputed_dataset_multitask(graph_dataset, best_state, init_params, args.skip_gnn)

    imp_acc_dict = testing_utils.measure_imp_accuracy(graph_dataset, df_imputed, logger=logger)

    # acc_results = testing_utils.test_acc(graph_dataset, gnn_model, prediction_model)
    # logger.update_dict('results',acc_results)
    logger.add_time('end_testing')
    logger.add_duration('start_testing', 'end_testing', 'testing_duration')
    logger.pprint()

    logger.save_obj(plot_figures=True)

    print(f"Testing took {time.time() - start:2f} seconds", )

    if args.save_imputed_df:
        fullname, ext = osp.splitext(args.dirty_dataset)
        basename = osp.basename(fullname)
        df_imp_fname = 'results/imputed_datasets/' + basename + '_imputed_grimp'
        if args.imputed_df_tag:
            df_imp_fname += f'_{args.imputed_df_tag}'
        df_imp_fname += ext
        df_imputed.to_csv(df_imp_fname, index=False)
        print(f'Imputed dataset saved in {df_imp_fname}')


def dump_graph(args):
    print("creating graph from scratch")

    original_data_file_name = args.ground_truth
    missing_value_file_name = args.dirty_dataset

    random_init = args.random_init

    if args.text_embs:
        node_mapping, ext_features = read_external_features(args.text_embs)
    else:
        ext_features = None
        node_mapping = None

    architecture = args.architecture
    if architecture == 'multitask':
        print('#### !!!!!! Creating multitask model.')
    elif architecture == 'multilabel':
        print('#### Creating base multilabel model.')
    else:
        raise ValueError(f'Unknown architecture {architecture}')

    graph_dataset = ImputationTripartiteGraphMultilabelClassifier(original_data_file_name,
                                                                  missing_value_file_name,
                                                                  random_init,
                                                                  node_mapping, ext_features, architecture=architecture,
                                                                  training_subset=args.training_subset)

    pickle.dump(graph_dataset, open(args.save_model_file_path, 'wb'))


def wrapper(args):
    # print (vars(args))
    # return
    logger.add_dict('parameters', vars(args))
    logger.add_run_name()
    logger.add_value('parameters', 'num_estimators', 0)
    logger.add_time('start_training')
    graph_dataset, best_state, init_params = run_training(args)
    logger.add_time('end_training')
    logger.add_duration('start_training', 'end_training', 'duration_training')

    run_testing(args, graph_dataset, best_state, init_params)
    logger.print_summary()
    logger.save_json()
    print(f'Completed run {logger.run_id}')


if __name__ == "__main__":
    args = parse_args()
    logger.add_dict('parameters', vars(args))
    logger.add_run_name()
    logger.add_value('parameters', 'num_estimators', 0)
    logger.add_time('start_training')
    graph_dataset, best_state, init_params = run_training(args)
    logger.add_time('end_training')
    logger.add_duration('start_training', 'end_training', 'duration_training')
    logger.print_summary()
    logger.save_json()
    run_testing(args, graph_dataset, best_state, init_params)

    print(f'Completed run {logger.run_id}')
