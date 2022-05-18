import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.nn import GraphConv
import pickle
from GRIMP.multilabel_graph_dataset import ImputationTripartiteGraphMultilabelClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class Predictor(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(Predictor, self).__init__()
        self.linear1 = nn.Linear(in_feats, in_feats)
        self.linear2 = nn.Linear(in_feats, out_feats)

    def forward(self, h):
        return self.linear2(self.linear1(h))

def prepare_labels(graph_dataset: ImputationTripartiteGraphMultilabelClassifier):
    labels = torch.Tensor(size=(graph_dataset.num_total_nodes, 1))
    labels[:graph_dataset.num_rows] = torch.zeros(size=(graph_dataset.num_rows, 1))
    # labels[graph_dataset.num_rows:graph_dataset.num_columns] = torch.ones(size=(graph_dataset.num_columns, 1))
    labels[graph_dataset.num_rows:graph_dataset.num_row_col_nodes] = torch.ones(size=(graph_dataset.num_columns, 1))
    labels[graph_dataset.num_row_col_nodes:] = 2 * torch.ones(size=(graph_dataset.num_total_nodes-graph_dataset.num_row_col_nodes,1))

    return labels.squeeze(1).to(int)


def plot_PCA(mat, labels, dim=2, title=None):
    pca = PCA(n_components=dim)
    pca_proj = pca.fit_transform(mat)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    colors = ['r', 'g', 'b']

    # ax.scatter(pca_proj[labels==0,0], pca_proj[labels==0,1], c=[colors[_] for _ in labels])
    ax.scatter(pca_proj[labels==0,0], pca_proj[labels==0,1], c='r')
    ax.scatter(pca_proj[labels==1,0], pca_proj[labels==1,1], c='g')
    ax.scatter(pca_proj[labels==2,0], pca_proj[labels==2,1], c='b')

    plt.legend(['rows', 'columns', 'values'])
    if title:
        plt.title(title)

    plt.show()

def train(g: ImputationTripartiteGraphMultilabelClassifier, model, predictor):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_train_acc = 0
    best_val_acc = 0
    best_test_acc = 0
    g.graph = dgl.add_self_loop(g.graph)
    g.graph.ndata['features'].requires_grad = True
    labels = prepare_labels(g)
    # train_mask = g.ndata['train_mask']
    # val_mask = g.ndata['val_mask']
    # test_mask = g.ndata['test_mask']

    model.eval()
    first_state = model(g.graph, g.graph.ndata['features'])
    model.train()

    plot_PCA(first_state.detach().numpy(), labels, title='Initial State')

    for e in range(200):
        # Forward
        node_reps = model(g.graph, g.graph.ndata['features'])

        logits = predictor(node_reps)
        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits, labels)

        # Compute accuracy on training/validation/test
        train_acc = (pred == labels).float().mean()
        # val_acc = (pred[val_mask] == labels[val_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_train_acc < train_acc:
            best_train_acc = train_acc
        # if best_val_acc < val_acc:
        #     best_val_acc = val_acc
        #     best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f},  (best {:.3f}),  (best {:.3f})'.format(
                e, loss, best_val_acc, best_test_acc))

    model.eval()
    final_state = model(g.graph, g.graph.ndata['features'])
    plot_PCA(final_state.detach().numpy(), labels, title='Final State')
    return g

if __name__ == '__main__':
    emb_file = 'adult-debug-pretrain-embdi'
    graph_dataset = pickle.load(open(emb_file, 'rb'))

    plot_PCA(graph_dataset.graph.ndata['features'].detach().numpy(), labels=prepare_labels(graph_dataset), title='Starting features')


    h_feats = 16
    starting_features = graph_dataset.graph.ndata['features']
    # Create the model with given dimensions
    model = GCN(graph_dataset.graph.ndata['features'].shape[1], h_feats, h_feats)
    predictor = Predictor(h_feats, 3)
    graph_dataset = train(graph_dataset, model, predictor)
    final_features = graph_dataset.graph.ndata['features']
    # torch.save(final_features, emb_file + '_mat.npy')


