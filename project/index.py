import torch, time, datetime
from torch_geometric.datasets import Planetoid
import numpy as np
import random
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch_geometric.nn as hnn
from torch_geometric.utils import degree
import torch.nn.functional as F
import os
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
        with open("{}/ind.{}.{}".format('Cora/raw', 'cora', names[i]), 'rb') as f:
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
    #  بدست آوردن لیست یال‌ها به همراه اضافه کردن خود راس به هر لیست
    #  به عبارتی داره هایپر یال‌ها رو بر اساس لیست درست می‌کنه
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
    def __init__(self, data, hyperedge_weight=None, hidden_representation_lenghtion=0, number_of_head=8):
        r"""
                Learning Hypergraph with/without hyperedge_weight
                :param data: Dataset data
                :param hyperedge_wieght: hyperedge_weight
                :return F.log_sonfmax(x, dim=1)
                """
        self.hidden_representation_lenght = hidden_representation_lenghtion
        self.number_of_head = number_of_head
        super(Hyper_Attention_Class, self).__init__()
        self.hconv1 = hnn.HypergraphConv(
            data.num_features,
            self.hidden_representation_lenght,
            use_attention=True,
            heads= self.number_of_head,
            concat=True,
            bias=True,
            dropout=0.6)
        self.hconv2 = hnn.HypergraphConv(
            self.hidden_representation_lenght * self.number_of_head,
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
            x = F.elu(self.hconv1(x, data.edge_index, hyperedge_weight.t()))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.hconv2(x, data.edge_index, hyperedge_weight.t())

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    # seed_value = 10000009
    # random.seed(seed_value)
    # os.environ['PYTHONHASHSEED'] = str(seed_value)
    # np.random.seed(seed_value)
    # torch.manual_seed(seed_value)

    # loading dataset
    datasetCora = Planetoid(root="Citeseer", name="Citeseer")
    data = datasetCora[0]

    sparse_H, H, ally = load_citation_data()

    # # ----------------------- create 2_step long neighborhood for Hypergraph ----------------------
    # # یافتن هایپریال با فاصله دو در دیتاست با استفاده از روش stationary distribution و ضرب دو ماتریس هایپر یال.
    # # نکته: البته ممکن است که این کار زمان محاسبات را بالا ببرد ولی فعلا برای کار راه‌اندازی خوب است.
    # # بعدا باید با استفاده از لیست‌ها که خود data.edge_index ها دارند این کار را انجام بدیم.
    # #
    # H_2 = torch.matmul(H, H)
    # H_2[H_2 >= 1] = 1
    # # convert Hypergraph incident matric to array with structure [2 * number of non-zero elements]
    # # # ساخت ماتریس هایپریال به صورت آرایه دو بعدی دو ضرب در تعداد المان‌های غیر صفر
    # b = 0
    # h_2 = torch.zeros(2, int(H_2.sum().item()))
    # for i in range(H_2.shape[0]):
    #     for j in range(H_2.shape[1]):
    #         if H_2[i, j]:
    #             h_2[0, b] = j # vertexes
    #             h_2[1, b] = i # hyperedges_index
    #             b += 1
    # # #  concatinate the hyperedges with distance 2 to data.edge_index
    # h_2 = h_2.long()
    # h_2_reverse = h_2
    # h_2_reverse[0], h_2_reverse[1] = h_2[1], h_2[0]
    # data.edge_index = torch.cat((h_2_reverse, h_2), 1)
    # # data.edge_index = torch.cat((data.edge_index, h_2_reverse), 1)
    # # num_distance_2_hyperedge = h_2.shape[1]
    # # print("data.edge_index.shape: ", data.edge_index.shape)
    # #

    # # --------------------------add each node to its hyperedge ----------------------------------
    # # در این مرحله هر نود را به هایپریال خود اضافه می‌کنیم
    tempedge = torch.zeros([2, 3327], dtype=torch.int64)
    for i in range(3327):
        tempedge[0, i], tempedge[1, i] = i, i
    data.edge_index = torch.cat((data.edge_index, tempedge), 1)
    # # num_immidiate_hyperedge = data.edge_index.shape[1]
    # # -------------------------------------------------------------------------------------------


    # # وزن تعریف کردن برای هایپر‌یال‌ها
    # hyperedge_weight = torch.ones([1, 2708], dtype=torch.float)
    # hyperedge_weight[0, 2708:5416] = 0.5
    # بدست آوردن درجه هر هایپریال به منظور درست کردن وزن هایپریا‌ل‌ها متناسب با درجه هر هایپریال
    #  و محاسبه وزن‌ها به صورت range normalization
    # hyperedge_degree = degree(data.edge_index[0], data.x.shape[0], data.x.dtype)
    # hyperedge_weight = hyperedge_degree - hyperedge_degree.min()
    # hyperedge_weight = hyperedge_weight / (hyperedge_degree.max() - hyperedge_degree.min())
    # hyperedge_weight_original = hyperedge_weight

    #
    # -------------------------------------------------------------------------------------------

    # -------------------learn with hypergraph attention network---------------------------------
    # model= Hypergraph_With_multiple_Hidden_Layer_Static()

    def train_model():
        model.train()
        optimizer.zero_grad()

        out = model(data,
                    hyperedge_weight=None)  # for hypergraph Hyperedge_weight=hyperedge_weight

        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        # loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

    def test_model():
        model.eval()
        logits, accs = model(data), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        # print(accs)
        return accs


    number_of_epochs = 10000
    number_of_trials = 1
    patience = 100
    last_val = 0
    trains_in_range_weight_iteration = 1
    # create table for plot accuracies
    plotvalues = np.zeros((number_of_epochs, 4))
    t0 = time.time()

    # hyperedge_weight -= hyperedge_weight_original
    for trains_in_in_specefic_weight in range(0, trains_in_range_weight_iteration):
        # hyperedge_weight += hyperedge_weight_original
        for trial in range(0, number_of_trials):
            # find the best accuracies during execution
            train_best = 0
            test_best = 0
            val_best = 0
            epoch_best = 0

            model = None
            # model = GAT_Net_Class()
            model = Hyper_Attention_Class(data=data,
                                          hyperedge_weight=None,
                                          hidden_representation_lenghtion=8,
                                          number_of_head=8)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

            for epoch in range(1, number_of_epochs):
                t = time.time()
                train_model()
                train_, val_, test_ = test_model()

                if (test_ > test_best):
                    epoch_best = epoch
                if(last_val >= val_):
                    patience -= 1
                    if patience == 0:
                        print('Early stop - best val accracy=%.4f and test accuracy=%.4f' % (val_best, test_best))
                        print('totoal time {}'.format(datetime.timedelta(seconds=time.time()-t0)))
                        break
                else:
                    patience = 100


                last_val = val_
                train_best = max(train_, train_best)
                test_best = max(test_, test_best)
                val_best = max(val_, val_best)

                plotvalues[epoch] = [epoch, train_, val_, test_]

                log = 'Trial:{} --> Epoch: {:03d} --> accuracy: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(trial, epoch, train_, val_, test_))
            print("The best accuracies:\n Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"
                  .format(train_best, val_best, test_best))

            # ----------------------------- Plot Code ----------------------------------
            print('totoal time {}'.format(datetime.timedelta(seconds=time.time()-t0)))
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
            plt.savefig('./PlotForConference/CiteSeer_self_loop_test_Trial_{}_best_test_{}_patience_100.png'
                        .format(trial, test_best), dpi=300)
            # plt.savefig('./Trial_{}_best_test_{}_hyperedge_weight {:.4f}.png'
            #             .format(trial, test_best, hyperedge_weight[0]), dpi=300)
            plt.show()
            # ------------------------------------------------------------------------