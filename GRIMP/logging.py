import pickle
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import os
import json

import pandas as pd
from abc import abstractmethod

RESULTS_PATH=osp.realpath('results')
PLOTS_PATH=osp.realpath('results/plots')
JSON_PATH=osp.join(RESULTS_PATH, 'json')
RUN_ID_PATH=osp.realpath('data/run_id')

class Logger:
    def __init__(self, file_path=None, run_id_path=RUN_ID_PATH,
                 results_path=RESULTS_PATH, plots_path=PLOTS_PATH):
        # self.run_id = 0
        self.run_id_path = run_id_path
        self.run_name = None
        self.results_path = results_path
        self.plots_path = plots_path
        self.run_id = self.find_latest_run_id()
        if file_path is None:
            self.obj = dict()
            self.obj['run_id'] = self.run_id
            self.obj['timestamps'] = dict()
            self.obj['durations'] = dict()
            self.obj['results'] = dict()
            self.obj['statistics'] = dict()
            self.add_time('logger_creation_time')
            os.makedirs('results', exist_ok=True)
            os.makedirs('results/plots', exist_ok=True)
        else:
            self.obj = pickle.load(open(file_path, 'rb'))
            self.run_id = self.obj['run_id']

    def find_latest_run_id(self):
        if osp.exists(self.run_id_path):
            with open(self.run_id_path, 'r') as fp:
                last_run_id = fp.read().strip()
                try:
                    run_id = int(last_run_id) + 1
                except ValueError:
                    raise ValueError(f'Run ID {last_run_id} is not a positive integer. ')
                if run_id < 0:
                    raise ValueError(f'Run ID {run_id} is not a positive integer. ')
            with open(self.run_id_path, 'w') as fp:
                fp.write(f'{run_id}')
        else:
            run_id = 0
            with open(self.run_id_path, 'w') as fp:
                fp.write(f'{run_id}')
        return run_id

    def add_dict(self, obj_name, obj):
        '''
        Add a new dictionary to the logger. `obj_name` is the key that will be used to store the object.

        :param obj_name: A string that will be used as key.
        :param obj: The dictionary to be added.
        :return:
        '''
        self.obj[obj_name] = dict()
        self.obj[obj_name].update(obj)

    def update_dict(self, obj_name, obj):
        self.obj[obj_name].update(obj)

    def add_value(self, obj_name, key, value):
        self.obj[obj_name][key] = value

    def add_run_name(self):
        basename = osp.basename(self.obj['parameters']['dirty_dataset'])
        name, ext = osp.splitext(basename)
        self.obj['parameters']['run_name'] = name
        self.run_name = name

    def get_value(self, obj_name, key):
        return self.obj[obj_name][key]

    def add_time(self, label):
        self.obj['timestamps'][label] = dt.datetime.now()

    def get_time(self, label, tformat=''):
        self.obj['timestamp'][label]

    def add_duration(self, label_start, label_end, label_duration):
        self.obj['durations'][label_duration] = (self.obj['timestamps'][label_end] - self.obj['timestamps'][label_start]).total_seconds()

    def save_obj(self, file_path=None):
        self.update_result_file()
        if file_path:
            pickle.dump(self.obj, open(file_path, 'wb'))
        else:
            file_path = osp.join(self.results_path, f'run_{self.run_id}.pkl')
            pickle.dump(self.obj, open(file_path, 'wb'))

    # @abstractmethod
    # def pprint(self):
    #     pass

    def __getitem__(self, item):
        return self.obj[item]

    def _print_features(self):
        if self.obj["parameters"]["external_feats"]:
            return self.obj["parameters"]["external_feats"]
        elif self.obj['parameters']['random_init']:
            return 'random_init'
        else:
            return 'basic_init'

    def pprint(self):
    #     Print basic info (method, dataset name, epochs, method, features type, results)
        res_string = f'{self.obj["parameters"]["link_predictor"]},' \
                     f'{self.obj["parameters"]["dirty_dataset"]},' \
                     f'{self.obj["statistics"]["training_columns"]},' \
                     f'{self.obj["statistics"]["node_features"]},' \
                     f'{self.obj["parameters"]["epochs"]},' \
                     f'{self.obj["parameters"]["predictor_structure"]},' \
                     f'{self.obj["parameters"]["gnn_structure"]},' \
                     f'{self.obj["results"]["imp_accuracy"]},'
        # print(res_string)
        return res_string

    @staticmethod
    def get_header():
        header = [
            'link_predictor',
            'dirty_dataset',
            'training_columns',
            'node_features',
            'epochs',
            'predictor_structure',
            'gnn_structure',
            'imp_accuracy'
        ]
        return header


    def print_selected(self, selected_dict):
        # TODO: this function is not done
        for obj_dict in selected_dict:
            selected_entries = selected_dict[obj_dict]
            for entry in selected_entries:
                val = self.obj[entry][entry]

    def update_result_file(self):
        if osp.exists(osp.join(self.results_path, 'results.csv')):
            with open(osp.join(self.results_path, 'results.csv'), 'a') as fp:
                fp.write(self.pprint())
        else:
            header = self.get_header()
            df = pd.DataFrame(columns=header)
            df.to_csv(osp.join(self.results_path, 'results.csv'), index=False)
            with open(osp.join(self.results_path, 'results.csv'), 'a') as fp:
                fp.write(self.pprint())

    def get_df_stats(self, df):
        self.obj['statistics']['num_rows'] = len(df)
        num_distinct_values = set(df.values.ravel().tolist())
        self.obj['statistics']['num_distinct_values'] = num_distinct_values
        num_missing_values = df.isna().sum().sum()
        self.obj['statistics']['num_missing_values'] = num_missing_values


class GrimpLogger(Logger):
    def pprint(self):
        self.summary = ''
        values = [
            self.run_id,
            'GRIMP-ML',
            self.obj['parameters']['dirty_dataset'],
            self.obj['parameters']['training_columns'],
            self.obj['statistics']['training_rows'],
            self.obj['parameters']['epochs'],
            self.obj['parameters']['predictor_structure'],
            self.obj['parameters']['gnn_structure'],
            self.obj['statistics']['unidirectional'],
            self.obj['parameters']['architecture'],
            self.obj['parameters']['no_relu'],
            self.obj['parameters']['no_sm'],
            self.obj['parameters']['k_strat'],
            self.obj['parameters']['learning_rate'],
            self.obj['parameters']['weight_decay'],
            self.obj['parameters']['module_aggr'],
            self.obj['parameters']['dropout_gnn'],
            self.obj['parameters']['dropout_clf'],
            self.obj['parameters']['flag_col'],
            self.obj['statistics']['comb_num'],
            self.obj['statistics']['comb_size'],
            '|'.join(self.obj['statistics']['node_features']),
            self.obj['parameters']['max_components'],
            self.obj['statistics']['training_columns'],
            # self.obj['statistics']['dtype_columns'],
            # 'num_fds',
            self.obj['statistics']['num_fds'],
            self.obj['statistics']['fd_strategy'],
            self.obj['statistics']['fd_path'],
            self.obj['statistics']['num_rows'],
            self.obj['statistics']['num_distinct_values'],
            self.obj['statistics']['num_missing_values'],
            self.obj['durations']['training_duration'],
            self.obj['results']['imp_accuracy'],
            self.obj['results']['tot_true'],
            self.obj['curves']['min'],
            self.obj['curves']['end'],
            self.obj['curves']['min_valid'],
            self.obj['curves']['end_valid'],
        ]
        if 'accuracy_dict' in self.obj['results']:
            for col in self.obj['results']['accuracy_dict']:
                values.append(self.obj['results']['accuracy_dict'][col])

        s = ','.join([str(_) for _ in values])
        # print(s)
        return s + '\n'

    def plot_curves(self):
        loss = self['curves']['loss']
        loss_valid = self['curves']['loss_valid']
        fig = plt.Figure()
        ax = fig.gca()
        p1,  = plt.plot(loss)
        p11, = plt.plot(loss_valid)
        p2,  = plt.plot(np.argmin(loss), np.min(loss), marker='o')
        p22, = plt.plot(np.argmin(loss_valid), np.min(loss_valid), marker='+')
        p3,  = plt.plot(len(loss), loss[-1], marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        short_name = osp.basename(self.obj['parameters']['dirty_dataset'])
        plt.title(f"Run {self.obj['run_id']} - Dataset {short_name}")
        plt.legend([p1, p11, p2, p22, p3], ['loss', 'loss_valid', 'min', 'min_valid', 'end'], loc='upper left')
        plt.savefig(osp.join(self.plots_path, f'{self.run_id}.png'))

    def print_summary(self):
        print(f"Run ID:{self.run_id}")
        print(f"Dataset: {self.obj['parameters']['dirty_dataset']}"),
        print(f"Training columns: {self.obj['parameters']['training_columns']}")
        print(f"Total epochs: {self.obj['parameters']['epochs']}")
        print(f"Architecture: {self.obj['parameters']['architecture']}")
        print(f"Loss function: {self.obj['parameters']['loss']}")
        print(f"Node features: {self.obj['statistics']['node_features']}")

    @staticmethod
    def get_header():
        header = [
            'run_id',
            'algorithm',
            'dirty_dataset',
            'training_columns',
            'training_rows',
            'epochs',
            'predictor_structure',
            'gnn_structure',
            'unidirectional',
            'architecture',
            'no_relu',
            'no_sm',
            'k_strat',
            'learning_rate',
            'weight_decay',
            'aggregation_function',
            'dropout_gnn',
            'dropout_clf',
            'cid_flag',
            'comb_num',
            'comb_size',
            'node_features',
            'max_components',
            'training_columns',
            'num_fds',
            'fd_strategy',
            'fd_path',
            'num_rows',
            'num_distinct_values',
            'num_missing_values',
            'training_duration',
            'imputation_accuracy',
            'total_correct_imputations',
            'min_loss',
            'final_loss',
            'min_valid_loss',
            'final_valid_loss',
        ]

        header = header + [f'imp_col_{i}' for i in range(1,21)]

        return  header

    def save_obj(self, file_path=None, plot_figures=False):
        super(GrimpLogger, self).save_obj(file_path)

        if plot_figures:
            self.plot_curves()

    def save_json(self):
        result_dict = {
            'dataset_name': self.run_name,
            'imputation_method': 'GRIMP',
            'run_params': self.obj['parameters'],
            'start_time': self.obj['timestamps']['start_training'].isoformat(),
            'end_time': self.obj['timestamps']['end_training'].isoformat(),
            'exec_time': self.obj['durations']['duration_training']
        }

        fpath = f'{self.run_name}_grimp.json'
        ofp = open(osp.join(JSON_PATH, fpath), 'w')
        json.dump(result_dict, ofp, indent=2)
        ofp.close()


def logging(parameters, results):
    logger = Logger()
    logger.add_dict('parameters', parameters)
    logger.add_dict('results', results)
