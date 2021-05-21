"""
created by weiyx15 @ 2019.1.4
Cora dataset interface
"""

import numpy as np
from config import get_config
from utils import edge_to_hyperedge
from utils.construct_hypergraph import generate_H_from_adjacency
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def encode_onehot(labels):
    classes = set(labels)

    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def encode_num(labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in
                    enumerate(classes)}
    labels_num = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_num


def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.mat(np.diag(r_inv))
    mx = r_mat_inv.dot(mx)
    return mx


def _view_index(idx, idx_map, edges):
    print(f'raw index: {idx}')
    print(f'index map: {idx_map}')
    nodes = [edge[0] for edge in edges]
    nodes += [edge[1] for edge in edges]
    nodes = list(set(nodes))
    print(len(nodes))
    nodes.sort()
    print(f'mapped index: {nodes}')


def load_cora_data(cfg, add_self_path=True):
    """
    cora data set with random split
    :param cfg:
    :param add_self_path:
    :return:
    """
    idx_features_labels = np.genfromtxt(cfg['cora_ft'], dtype=np.dtype(str))
    fts = idx_features_labels[:, 1:-1].astype(np.float32)
    fts = normalize_features(fts)
    lbls = encode_num(idx_features_labels[:, -1])

    n_train, n_val, n_test = cfg['train'], cfg['val'], cfg['test']
    idx_train = list(range(n_test + n_val, n_test + n_val + n_train))
    idx_val = list(range(n_test, n_test + n_val))
    idx_test = list(range(n_test))

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(cfg['cora_graph'], dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    if add_self_path:
        edges_list = edges.tolist()
        self_edges = np.array([[i, i] for i in range(fts.shape[0]) if [i, i] not in edges_list])
        edges = np.vstack((edges, self_edges))
        del edges_list

    node_dict, edge_dict = edge_to_hyperedge(edges)
    n_category = lbls.max() + 1
    return fts, lbls, idx_train, idx_val, idx_test, n_category, node_dict, edge_dict


def parse_index_file(filename):
    """
    Copied from gcn
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))# sum os each paper citation mustafa
    r_inv = np.power(rowsum, -1).flatten()# Normalized rowsum mustafa
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def load_citation_data(cfg):
    """
    Copied from gcn
    citeseer/cora/pubmed with gcn split
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = [] # list
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(cfg['citation_root'], cfg['activate_dataset'], names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(cfg['citation_root'], cfg['activate_dataset']))
    test_idx_range = np.sort(test_idx_reorder)

    if cfg['activate_dataset'] == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)
    features = features.todense()#convert lil_matrix to dense(regular) matrix mustafa

    G = nx.from_dict_of_lists(graph)

    # ======================== My code ================
    #  this code is added by my self to manipulate Hyperedge matrix for hypergraph attention network
    # call by myself

    H = generate_H_from_adjacency(data_graph=graph, k_adjacent_distance=1)
    print(H)
    # -----------------------------------------------
    #  بدست آوردن لیست یال‌ها به همراه اضافه کردن خود راس به هر لیست
    #  به عبارتی داره هایپر یال‌ها رو بر اساس لیست درست می‌کنه
    try:
        edge_list = G.adjacency_list()
    except :
        edge_list = [[]]*len(G.nodes)
        for idx, neigs in G.adjacency():
            edge_list[idx] = list(neigs.keys())

    degree = [0] * len(edge_list)
    if cfg['add_self_loop']:
        for i in range(len(edge_list)):
            edge_list[i].append(i)
            degree[i] = len(edge_list[i])
    # -----------------------------------------

    max_deg = max(degree)
    mean_deg = sum(degree) / len(degree)
    print(f'max degree: {max_deg}, mean degree:{mean_deg}')

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]     # one-hot labels
    n_sample = labels.shape[0]
    n_category = labels.shape[1]
    lbls = np.zeros((n_sample,))
    if cfg['activate_dataset'] == 'citeseer':
        n_category += 1                                         # one-hot labels all zero: new category
        for i in range(n_sample):
            try:
                lbls[i] = np.where(labels[i]==1)[0]                     # numerical labels
            except ValueError:                              # labels[i] all zeros
                lbls[i] = n_category + 1                        # new category
    else:
        for i in range(n_sample):
            lbls[i] = np.where(labels[i]==1)[0]                     # numerical labels

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    return features, lbls, idx_train, idx_val, idx_test, n_category, edge_list, edge_list, H


if __name__ == '__main__':
    cfg = get_config('../config/config_cora.yaml')
    if cfg['standard_split']:
        load_citation_data(cfg)
    else:
        load_cora_data(cfg)
