import torch
from torch_geometric.datasets import Planetoid
import numpy as np
import random
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch_geometric.nn as hnn
from torch_geometric.utils import degree
import torch.nn.functional as F
import matplotlib.pyplot as plt

def generate_H_from_adjacency(data_graph, k_adjacent_distance=1):
    #  create Hyperedge longtensor matrix
    temphyperedge = np.ndarray((2708, 2708))
    # temphyperedge = data_graph + temphyperedge + np.eye(2708)

    for i in range(2708):
        for j in range(len(data_graph[i])):
            # print("temphyperedge [{},{}] is set to 1".format(i, data_graph[i][j]))
            temphyperedge[i, (data_graph[i][j])] = 1
    temphyperedge = temphyperedge + np.eye(2708)
    hyperedge = torch.LongTensor(temphyperedge)

    return hyperedge

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))  # sum os each paper citation mustafa
    r_inv = np.power(rowsum, -1).flatten()  # Normalized rowsum mustafa
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def parse_index_file(filename):
    """
    Copied from gcn
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_citation_data():

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []  # list
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format('Cora/raw/', 'cora', names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format("Cora/raw", "cora"))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)
    features = features.todense()  # convert lil_matrix to dense(regular) matrix mustafa

    G = nx.from_dict_of_lists(graph)

    # ======================== My code ================
    #  this code is added by my self to manipulate Hyperedge matrix for hypergraph attention network
    # call by myself

    H = generate_H_from_adjacency(data_graph=graph, k_adjacent_distance=1)
    sparse_H = H.to_sparse()
    print(sparse_H)
    # sparse_H = sparse.csr_matrix(H)
    # print(H)
    # -----------------------------------------------
    #  ???????? ?????????? ???????? ????????????? ???? ?????????? ?????????? ???????? ?????? ?????? ???? ???? ????????
    #  ???? ???????????? ???????? ?????????? ????????????? ???? ???? ???????? ???????? ???????? ?????????????
    try:
        edge_list = G.adjacency_list()
    except:
        edge_list = [[]] * len(G.nodes)
        for idx, neigs in G.adjacency():
            edge_list[idx] = list(neigs.keys())

    degree = [0] * len(edge_list)
    # if cfg['add_self_loop']
    if True:
        for i in range(len(edge_list)):
            edge_list[i].append(i)
            degree[i] = len(edge_list[i])
    # -----------------------------------------

    # max_deg = max(degree)
    # mean_deg = sum(degree) / len(degree)
    # print(f'max degree: {max_deg}, mean degree:{mean_deg}')

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # one-hot labels
    n_sample = labels.shape[0]
    n_category = labels.shape[1]
    lbls = np.zeros((n_sample,))

    for i in range(n_sample):
        lbls[i] = np.where(labels[i] == 1)[0]  # numerical labels

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    return sparse_H, H, ally


class GAT_Net_Class(torch.nn.Module):
    def __init__(self):
        super(GAT_Net_Class, self).__init__()
        self.conv1 = hnn.GATConv(datasetCora.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = hnn.GATConv(8 * 8, datasetCora.num_classes, heads=1, concat=True, dropout=0.6)

    def forward(self):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


class Hyper_Attention_Class(torch.nn.Module):
    def __init__(self, data, hyperedge_weight):
        r"""
                Learning Hypergraph with/without hyperedge_weight
                :param data: Dataset data
                :param hyperedge_wieght: hyperedge_weight
                :return F.log_sonfmax(x, dim=1)
                """
        super(Hyper_Attention_Class, self).__init__()
        self.hconv1 = hnn.HypergraphConv(
            data.num_features,
            8,
            use_attention=True,
            heads=8,
            concat=True,
            bias=True,
            dropout=0.6
        )
        self.hconv2 = hnn.HypergraphConv(
            8 * 8,
            datasetCora.num_classes,
            use_attention=False,
            heads=1,
            concat=True,
            bias=True,
            dropout=0.6
        )

    def forward(self, data, hyperedge_weight=None):

        if hyperedge_weight is None:
            x = F.dropout(data.x, p=0.6, training=self.training)
            x = F.elu(self.hconv1(x, data.edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.hconv2(x, data.edge_index)
        else:
            x = F.dropout(data.x, p=0.6, training=self.training)
            x = F.elu(self.hconv1(x, data.edge_index, hyperedge_weight))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.hconv2(x, data.edge_index, hyperedge_weight)

        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    # random.seed(10000009)

    # loading dataset
    datasetCora = Planetoid(root="Cora", name="Cora")
    data = datasetCora[0]

    sparse_H, H, ally = load_citation_data()


    # --------------------------add each node to its hyperedge ----------------------------------
    # ???? ?????? ?????????? ???? ?????? ???? ???? ???????????????? ?????? ?????????? ???????????????
    tempedge = torch.zeros([2, 2708], dtype=torch.int64)
    for i in range(2708):
        tempedge[0, i], tempedge[1, i] = i, i
    data.edge_index = torch.cat((data.edge_index, tempedge), 1)
    num_immidiate_hyperedge = data.edge_index.shape[1]
    # -------------------------------------------------------------------------------------------
    # # ?????? ?????????? ???????? ???????? ??????????????????????????
    # ???????? ?????????? ???????? ???? ???????????????? ???? ?????????? ???????? ???????? ?????? ?????????????????????????? ???????????? ???? ???????? ???? ????????????????
    #  ?? ???????????? ????????????? ???? ???????? range normalization and Z-score
    # hyperedge_degree = degree(data.edge_index[0], data.x.shape[0], data.x.dtype)
    # hyperedge_weight = hyperedge_degree - hyperedge_degree.min()
    # hyperedge_weight = hyperedge_weight / (hyperedge_degree.max() - hyperedge_degree.min())
    # hyperedge_weight += 3
    # define hyperedge_weight_static for hyperedge_weight distance 2
    # hyperedge_weight = torch.ones([1, 2708], dtype=torch.float)
    # hyperedge_weight[0, 2708:5416] = 0.5
    # hyperedge_weight += 3
    # define hyperedge_weight_static for hyperedge_weight distance 2
    # hyperedge_weight = torch.ones([1, 2708], dtype=torch.float)
    # hyperedge_weight[0, 2708:5416] = 0.5
    

    # ----------------------- create 2_step long neighborhood for Hypergraph ----------------------
    # ?????????? ???????????????? ???? ?????????? ???? ???? ???????????? ???? ?????????????? ???? ?????? stationary distribution ?? ?????? ???? ???????????? ?????????? ??????.
    # ????????: ?????????? ???????? ?????? ???? ?????? ?????? ???????? ?????????????? ???? ???????? ???????? ?????? ???????? ???????? ?????? ????????????????????? ?????? ??????.
    # ???????? ???????? ???? ?????????????? ???? ??????????????? ???? ?????? data.edge_index ???? ?????????? ?????? ?????? ???? ?????????? ????????.
    #
    H_2 = torch.matmul(H, H)
    H_2[H_2 >= 1] = 1
    # # # # convert Hypergraph incident matric to array with structure [2 * number of non-zero elements]
    # # # # ???????? ???????????? ???????????????? ???? ???????? ?????????? ???? ???????? ???? ?????? ???? ?????????? ??????????????????? ?????? ??????
    b = 0
    h_2 = torch.zeros(2, int(H_2.sum().item()))
    for i in range(H_2.shape[0]):
        for j in range(H_2.shape[1]):
            if H_2[i, j]:
                h_2[0, b] = i
                h_2[1, b] = j
                b += 1
    # ???????? ?????????? ?????? ????????????????????????? ???? ?????????? ????
    #  concatinate the hyperedges with distance 2 to data.edge_index
    h_2 = h_2.long()
    num_distance_2_hyperedge = h_2.shape[1]
    data.edge_index = torch.cat((data.edge_index, h_2), 1)
    print("data.edge_index.shape: ", data.edge_index.shape)

    """
        ???? ?????????? ?????????? ?????? ?????????? ????????????????????? ???? ?????????? ???????? ????????   
        ???? ???? ?????????? ????????????????????????? ???? ???????? ???? ???? ???????? ????????????
        ???? ????????????????????????? ???? ???????? ???? ?????? ?????????? ??.?? ?? ???????? ????????????????????????? ???? ?????????? ???? ?????? ??.?? ?????????? ??????????????? ?? ???????? ???? ?????????? ???????????????
        ???????????? ???? ???????? ???????? ??????????
        ????????: ???????? ?????????????????????????? ???? ???????? ???? ???? ?????????? ???? ???????? ???? ?????? ?????????? ?????????? ?????? 
        ???? ???? ?????????? ???? ?????????? ???????? ???????????? ?????? ???? ???????? ?????????? ???? ???????? ?????? ????????????????????????? ???? ?????????? ???? ??
        ?? ???????? ?????????? ?????????? ???????? ????????????????????????? ???? ?????????? ???? ???????????????.
    """
    flexible_hyperedge_weight_1 = 1
    hyperedge_weight = torch.ones([5416], dtype=torch.float)
    # hyperedge_weight = torch.ones([5416], dtype=torch.float)
    # hyperedge_weight[0:2078] = 0.8     #?????? ????????????????????????? ???? ?????????? ???? 
    # hyperedge_weight[2708:5416] = 0.2  #?????? ????????????????????????? ???? ?????????? ????
    hyperedge_degree_1 = degree(data.edge_index[0,:13264], data.x.shape[0], data.x.dtype)
    hyperedge_degree_2 = degree(data.edge_index[0,13264:], data.x.shape[0], data.x.dtype)
    # hyperedge_degree = torch.cat((hyperedge_degree_1, hyperedge_degree_2), dim=0)
    hyperedge_weight_1 = hyperedge_degree_1 - hyperedge_degree_1.min()
    hyperedge_weight_2 = hyperedge_degree_2 - hyperedge_degree_2.min()
    # hyperedge_weight = hyperedge_degree - hyperedge_degree.min()
    hyperedge_weight_1 = hyperedge_weight_1 / (hyperedge_degree_1.max() - hyperedge_degree_1.min())
    hyperedge_weight_2 = hyperedge_weight_2 / (hyperedge_degree_2.max() - hyperedge_degree_2.min())
    # in this type of hyperedge wieghts we use statistical weight for distance 1 and distance 2

    # #
    #
    # -------------------------------------------------------------------------------------------


    # -------------------learn with hypergraph attention network---------------------------------

    # def train_model():
    #     model.train()
    #     optimizer.zero_grad()
    #     out = model(data, hyperedge_weight=hyperedge_weight) # for hypergraph Hyperedge_weight=hyperedge_weight
    #     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #     loss.backward()
    #     optimizer.step()

    def test_model():
        model.eval()
        logits, accs = model(data), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        # print(accs)
        return accs

    # define number of epochs and trials respectively in following two next lines
    number_of_trials = 20
    number_of_epochs = 300

    # create table for plot accuracies
    plotvalues = np.zeros((number_of_epochs, 4))

    enumerate_hyperedge_weight = 5
    for run_program_for_different_weights in range(enumerate_hyperedge_weight):
        hyperedge_weight_1 *= flexible_hyperedge_weight_1
        hyperedge_weight_2 *= 1- flexible_hyperedge_weight_1
        hyperedge_weight  = torch.cat((hyperedge_weight_1, hyperedge_weight_2), dim=0)
        flexible_hyperedge_weight_1 -= 0.2 

        for trial in range(1, number_of_trials):
            # find the best accuracies during execution
            train_best = 0
            test_best = 0
            val_best = 0
            epoch_best = 0
            # ------------------------------------------
            
            model = None
        
            model = Hyper_Attention_Class(data=data, hyperedge_weight=hyperedge_weight)  # model = GAT_Net_Class()    
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

            for epoch in range(1, number_of_epochs):
                # train_model() -------------- pass this module to the main builting block
                model.train()
                optimizer.zero_grad()
                out = model(data, hyperedge_weight=hyperedge_weight) # for hypergraph Hyperedge_weight=hyperedge_weight
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                # ------------------------------------  

                train_, val_, test_ = test_model()

                if(test_ > test_best):
                    epoch_best = epoch

                # calculate best results
                train_best = max(train_, train_best)
                test_best = max(test_, test_best)
                val_best = max(val_, val_best)
                # ----------------------

                plotvalues[epoch] = [epoch, train_, val_, test_]

                log = 'Trial:{} --> Epoch: {:03d} --> accuracy: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(trial, epoch, train_, val_, test_))
            
            print("The best accuracies:\n Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"
                .format(train_best, val_best, test_best))
            # ----------------------------- Plot Code ----------------------------------
            plt.figure(dpi=300)
            plt.rc('ytick', labelsize=6)
            plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
            plt.yticks(np.arange(0, 1, step=0.02))
            line1 = plt.plot(plotvalues[:, 0], plotvalues[:, 1], 'g-', label='Train acc')
            line1 = plt.plot(plotvalues[:, 0], plotvalues[:, 2], 'r-', label='Validation acc')
            line1 = plt.plot(plotvalues[:, 0], plotvalues[:, 3], label='Test acc')

            # line1.set
            plt.title('Accuracies')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                    ncol=2, mode="expand", borderaxespad=0.)
            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            text = "epoch={}, accuracy=%{:.3f}".format(epoch_best, (test_best * 100))
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
            arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
            kw = dict(xycoords='data', textcoords="axes fraction",
                    arrowprops=arrowprops, bbox=bbox_props, ha="right", va="bottom")
            plt.annotate(text, xy=(epoch_best, test_best), xytext=(0.94, 0.96), **kw)

            plt.grid()
            plt.savefig('./plots/shabankhah/plot:{}-{}___with_weightOnImmidiateWeight_{:.2}_bestTest_%{:.2f}_bestval_%{:.2f}.png'
                        .format(run_program_for_different_weights, trial, flexible_hyperedge_weight_1, test_best * 100, val_best * 100), dpi=300)
            plt.show()
            # ------------------------------------------------------------------------
