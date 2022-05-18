#GiG
import os

import pandas as pd
import numpy as np
import torch
import dgl
from dgl.data import DGLDataset



class ImputationTripartiteGraphDatasetEdgePrediction(DGLDataset):
    def __init__(self, original_data_file_name, missing_value_file_name):
        self.original_data_file_name = original_data_file_name
        self.missing_value_file_name = missing_value_file_name
        self.num_rows = self.num_columns = 0
        # num(rows) + num(columns)
        self.num_row_col_nodes = 0
        # num(rows) + num(columns) + num(distinct cell values in original_data_file_name)
        self.num_total_nodes = 0
        super().__init__(name='EmbDI Graph')

    #This function loads the two data frames, does some bookkeeping and computes basic stats
    def load_and_compute_stats(self):
        #Read everything as a string
        self.df_orig = pd.read_csv(self.original_data_file_name, dtype=str)
        self.df_missing = pd.read_csv(self.missing_value_file_name, dtype=str)

        if self.df_orig.shape != self.df_missing.shape:
            raise Exception("Error: Input files do not have same number of rows and columns")

        #Convert every attribute to string and treat it as categorical
        #Redundant given the dtype in read_csv but still
        for col in self.df_orig.columns:
            self.df_orig[col] = self.df_orig[col].astype(str)
        for col in self.df_missing.columns:
            self.df_missing[col] = self.df_missing[col].astype(str)

        #Using dtype in read_csv makes missing value as a string 'nan'. So replace it appropriately to nan
        self.df_missing.replace('nan', np.nan, inplace=True)

        #Node id semantics: let n, m, l be number of rows, columns and distinct cell values
        #then node ids 0 to n-1 correspond to rows,
        # node ids n to n+m-1 correspond to columns
        # and n+m to n+m+l-1 correspond to cells
        self.num_rows, self.num_columns = self.df_orig.shape
        self.num_row_col_nodes = self.num_rows + self.num_columns

        self.distinct_value_set = {self.df_orig.iloc[row, col] for row in range(self.num_rows) for col in range(self.num_columns)}
        self.distinct_value_dict = dict( (v, i+self.num_row_col_nodes) for (i, v) in enumerate(sorted(self.distinct_value_set)))
        self.distinct_value_dict_reverse = dict( (i+self.num_row_col_nodes, v) for (i, v) in enumerate(sorted(self.distinct_value_set)))

        #For each distinct value associate with it the column in which it is found
        #if a value V is present in two columns A and B, then it will be associated with the latest column
        self.cell_to_dict = {self.df_orig.iloc[row, col]: col for row in range(self.num_rows) for col in range(self.num_columns)}

        self.column_dict = dict( (col, self.num_rows+i) for (i, col) in enumerate(self.df_orig.columns))
        self.column_dict_reverse = dict( (self.num_rows+i, col) for (i, col) in enumerate(self.df_orig.columns))

        self.num_total_nodes = self.num_row_col_nodes + len(self.distinct_value_set)

        #Aggregate for each column and then aggregate for entire data frame
        self.num_missing_values = self.df_missing.isna().sum().sum()
        self.num_non_missing_values = len(self.distinct_value_set)


    def create_graph(self):
        #Create empty graph
        self.graph = dgl.graph(data=[])

        #Create num_total_nodes isolated nodes
        self.graph.add_nodes(self.num_total_nodes)

        #Create edges
        #Note that we use df_missing for creating the edges
        #So cells corresponding to missing values will be isolated
        num_edges = 0
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                if pd.isnull(self.df_missing.iloc[row, col]) == False:
                    num_edges = num_edges + 2

        #DGL expects the edges to be specified as two tensors containing start and end nodes.
        start_nodes = torch.zeros(num_edges, dtype=torch.int64)
        end_nodes = torch.zeros(num_edges, dtype=torch.int64)

        index = 0
        for row in range(self.num_rows):
            row_node_id = row
            for col in range(self.num_columns):
                col_node_id = col + self.num_rows
                if pd.isnull(self.df_missing.iloc[row, col]) == False:
                    cell_node_id = self.distinct_value_dict[self.df_missing.iloc[row, col]]
                    start_nodes[index] = row_node_id
                    end_nodes[index] = cell_node_id
                    start_nodes[index+1] = col_node_id
                    end_nodes[index+1] = cell_node_id
                    index = index + 2

        self.graph.add_edges(start_nodes, end_nodes)

        #Make the graph undirected
        self.graph = dgl.add_reverse_edges(self.graph)

    def compute_node_features(self):
        #ndata stores various features for each node.
        #as a start, lets create a single feature variable dubbed "features"
        #TODO: this is only a start. a node feature of dimension self.num_columns is typically too low
        # typically we need in the order of 100s
        # We need to add additional features  that injects FD related information
        self.num_features = self.num_columns
        self.graph.ndata['features'] = torch.zeros( (self.num_total_nodes, self.num_columns) )

        #Features for row nodes = it is connected to each column
        self.graph.nodes[range(0, self.num_rows)].data['features'] = torch.ones(self.num_rows, self.num_columns)

        #Features for column nodes = it is connected to only itself
        self.graph.nodes[range(self.num_rows, self.num_rows+self.num_columns)].data['features'] = torch.eye(self.num_columns)

###############TODO: more expensive and probably better set of features
#        #For cell nodes, create an edge between that node and the column in which it was present
#        for row in range(self.num_rows):
#            for col in range(self.num_columns):
#                col_node_id = col + self.num_rows
#                if pd.isnull(self.df_missing.iloc[row, col]) == False:
#                    cell_node_id = self.val2idx[self.df_missing.iloc[row, col]]
###############This is the expensive part .... figure out a way to to modify ndata inplace
######################perhaps by first using a tensor matrix and then set to  ndata['features'] at the end.
#                    temp = torch.zeros(1, self.num_columns)
#                    temp[0][col] = 1.0
#                    self.graph.nodes[cell_node_id].data['features'] = temp
        #Features for cell nodes = it is connected to each column
        self.graph.nodes[range(self.num_row_col_nodes, self.num_total_nodes)].data['features'] = torch.ones(self.num_total_nodes-self.num_row_col_nodes, self.num_columns)


    #All the existing edges will be considered as positive examples
    #We now create a negative example of the same size
    #For now negative sampling works as follows:
    #for a row/column node, we randomly choose a cell node
    #for a cell node, we randomly choose a row node
    #we verify to make sure that edge does not already exists
    def get_negative_edges_for_training(self):
        #DGL expects the edges to be specified as two tensors containing start and end nodes.
        start_nodes = torch.zeros(self.num_total_nodes, dtype=torch.int32)
        end_nodes = torch.zeros(self.num_total_nodes, dtype=torch.int32)

        #IMPORTANT the lower bound is inclusive and upper bound is exclusive
        row_id_start, row_id_end = 0, self.num_rows
        col_id_start, col_id_end = self.num_rows, self.num_rows + self.num_columns
        cell_id_start, cell_id_end = self.num_rows + self.num_columns, self.num_total_nodes

        row_id_range = range(row_id_start, row_id_end)
        col_id_range = range(col_id_start, col_id_end)
        cell_id_range = range(cell_id_start, cell_id_end)

        #Precompute some random cell ids
        random_cell_ids = torch.randint(low=cell_id_start, high=cell_id_end, size=(self.num_total_nodes,))
        random_row_ids = torch.randint(low=row_id_start, high=row_id_end, size=(self.num_total_nodes,))
        random_cell_id_index = 0
        random_row_id_index = 0

        for node_id in range(self.num_total_nodes):
            #Corresponds to a valid column id: pick a random cell id for negative edge
            if row_id_start <= node_id < row_id_end:
                start_nodes[node_id] = node_id
                end_nodes[node_id] = random_cell_ids[random_cell_id_index]
                random_cell_id_index += 1
            #Corresponds to a valid column id: pick a random cell id for negative edge
            elif col_id_start <= node_id < col_id_end:
                start_nodes[node_id] = node_id
                end_nodes[node_id] = random_cell_ids[random_cell_id_index]
                random_cell_id_index += 1
            #Corresponds to a valid column id: pick a random ROW  id for negative edge
            #TODO: test other things
            elif cell_id_start <= node_id < cell_id_end:
                start_nodes[node_id] = node_id
                end_nodes[node_id] = random_row_ids[random_row_id_index]
                random_row_id_index += 1
            else:
                print("Error: column id seems erroneous %d " % node_id)


        return start_nodes, end_nodes

    #This function creates a positive graph for testing.
    #Here we treat it as a simple edge link prediction
    #So instead of creating it as a triplet, we only consider the edge between row_id, cell_id

    def get_positive_edges_for_testing(self):
        #DGL expects the edges to be specified as two tensors containing start and end nodes.
        start_nodes = torch.zeros(self.num_missing_values, dtype=torch.int32)
        end_nodes = torch.zeros(self.num_missing_values, dtype=torch.int32)

        index = 0

        for row in range(self.num_rows):
            row_node_id = row
            for col in range(self.num_columns):
                col_node_id = col + self.num_rows
                if pd.isnull(self.df_missing.iloc[row, col]) == True:
                    #NOTE: we change again from df_missing to df_orig to get the correct value
                    cell_node_id = self.distinct_value_dict[self.df_orig.iloc[row, col]]
                    start_nodes[index] = row_node_id
                    end_nodes[index] = cell_node_id
                    index = index + 1

        return start_nodes, end_nodes

    #The negative graph is obtained my selecting any value other than the correct one.
    #For e.g. (Qatar, NULL) is a tuple with missing value
    #Then (Qatar, Doha) is a positive example while (Qatar, Rome) is a negative value
    def get_negative_edges_for_testing(self):
        #DGL expects the edges to be specified as two tensors containing start and end nodes.
        start_nodes = torch.zeros(self.num_missing_values, dtype=torch.int32)
        end_nodes = torch.zeros(self.num_missing_values, dtype=torch.int32)

        cell_id_start, cell_id_end = self.num_rows + self.num_columns, self.num_total_nodes
        #Precompute some random cell ids
        random_cell_ids = torch.randint(low=cell_id_start, high=cell_id_end, size=(self.num_total_nodes,))

        index = 0
        random_cell_id_index = 0
        for row in range(self.num_rows):
            row_node_id = row
            for col in range(self.num_columns):
                col_node_id = col + self.num_rows
                if pd.isnull(self.df_missing.iloc[row, col]) == True:
                    #NOTE: we change again from df_missing to df_orig to get the correct value
                    cell_node_id = self.distinct_value_dict[self.df_orig.iloc[row, col]]
                    start_nodes[index] = row_node_id
                    end_nodes[index] = random_cell_ids[random_cell_id_index]
                    index = index + 1
                    random_cell_id_index = random_cell_id_index + 1
        return start_nodes, end_nodes



    #This creates four "graphs" on which training and evaluation is done.
    #train_{pos,neg}, test_{pos,neg}
    #train_pos are all edges in the graph (excluding the missing values)
    #train_neg are all edges not present in the graph  (excluding the missing values)
    #test_pos are all the edges for the missing values
    #test_neg is currently treated as optional
    #TODO: find a semantics for that
    def create_train_test_positive_negative_graphs(self):
        #For now, creating a new graph - but could also use self.graph itself
        print("creating train_positive_samples")
        train_pos_u, train_pos_v = self.graph.edges()
        self.train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=self.graph.number_of_nodes())

        print("creating train_negative_samples")
        start_nodes, end_nodes = self.get_negative_edges_for_training()
        self.train_neg_g = dgl.graph((start_nodes, end_nodes), num_nodes=self.graph.number_of_nodes())

        print("creating test_positive_samples")
        start_nodes, end_nodes = self.get_positive_edges_for_testing()
        self.test_pos_g = dgl.graph((start_nodes, end_nodes), num_nodes=self.graph.number_of_nodes())


        print("creating test_negative_samples")
        start_nodes, end_nodes = self.get_negative_edges_for_testing()
        self.test_neg_g = dgl.graph((start_nodes, end_nodes), num_nodes=self.graph.number_of_nodes())




    def process(self):
        print("Loading and computing basic stats")
        self.load_and_compute_stats()
        print("Creating graph structure")
        self.create_graph()
        print("Computing graph features")
        self.compute_node_features()
        print("Creating graphs for train test")
        self.create_train_test_positive_negative_graphs()

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
