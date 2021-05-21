"""
transform graphs (represented by edge list) to hypergraph (represented by node_dict & edge_dict)
"""
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances as cos_dis, euclidean_distances
from sklearn.cluster import KMeans
from models.layers import SampledGraphConvolution
from torch_geometric.nn.conv import hypergraph_conv
import collections


def edge_to_hyperedge(edges):
    """
    transform edges to hyperedges
    For hyperedges constructed by existed graph edges, hyperedge_id = centroid_node_id
    :param edge_list: list of edges (numpy array)
    :return: node_dict: edges containing the node
    :return: edge_dict: nodes contained in the edge
    """
    edge_list = [list() for i in range(edges.max()+1)]
    # node_cited = set()
    # node_list = [list() for i in range(edges.max()+1)]
    for edge in edges:
        # edge[0]: paper cited; edge[1]: paper citing
        edge_list[edge[0]].append(edge[1])
        edge_list[edge[1]].append(edge[0])
        # node_cited.add(edge[1])
    # print(len(node_cited))
    node_list = edge_list
    return node_list, edge_list


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                H = np.hstack((H, h))
    return H


def construct_H_with_KNN(X, K_neigs=[10], is_probH=False, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = cos_dis(X)
    H = None
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        H = hyperedge_concat(H, H_tmp)
    return H


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H
#================================================================
#  defined by Mustafa Mohammadi
# create H from Adjacency
def generate_H_from_adjacency(data_graph, k_adjacent_distance =1):
    r"""
    calculate H for Hypergraph using Adjacency Learning
    H is N * M with N vertices and M edges
    :param data_graph
    :param k_adjacent_distance
    :return H with course of adjacency matrix
    H is long tensor
    """
    r"""
    print("generate H from Adjacency")
    print("\n\n\n")
    print("node 0 : ", data_graph[0])
    # print(graph[0][0])
    # print(graph[0][1])
    # print(graph[0][2])
    print("node 633 : ", data_graph[633])
    print("node 1862 : ", data_graph[1862])
    print("node 2582 : ", data_graph[2582])
    print("\n\nall adjacent of node 0 with distance 2:\n\n")
    a = data_graph[0]

    for i in a:
        print(data_graph[i])

    data_graph = data_graph + np.eye(len(data_graph))
    print("len(data_graph): ", len(data_graph))
    H = []
    for i in range(len(data_graph)):
        data_graph[i].append(i)

    # gen_dict = collections.defaultdict(data_graph)

    print("type of data_graph: ", type(data_graph))
    """
    #  create Hyperedge longtensor matrix
    temphyperedge = np.ndarray((2708, 2708))
    for i in range(2708):
        for j in range(len(data_graph[i])):
            print("temphyperedge [{},{}] is set to 1".format(i, data_graph[i][j]))
            temphyperedge[i, (data_graph[i][j])] = 1
    temphyperedge = temphyperedge + np.eye(2708)
    hyperedge = torch.LongTensor(temphyperedge)

    return hyperedge

def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def construct_G_from_fts(Xs, k_neighbors):
    """
    generate G from concatenated H from list of features
    :param Xs: list of features
    :param k_neighs: list of k
    :return: numpy array
    """
    Hs = [construct_H_with_KNN(Xs[i], [k_neighbors[i]]) for i in range(len(Xs))]
    H = np.concatenate(Hs, axis=1)
    G = generate_G_from_H(H)
    return G


def H_to_node_edge_dict(H):
    H = np.array(H, dtype=np.int)
    row, col = np.where(H==1)
    n_node, n_edge = H.shape[0], H.shape[1]
    node_dict = [list() for i in range(n_node)]
    edge_dict = [list() for i in range(n_edge)]
    for i in range(row.size):
        node_dict[row[i]].append(col[i])
        edge_dict[col[i]].append(row[i])
    return node_dict, edge_dict


def _construct_edge_list_from_distance(X, k_neigh):
    """
    construct edge_list (numpy array) from kNN distance for single modality
    :param X -> numpy array: feature
    :param k_neigh -> int: # of neighbors
    :return: N * k_neigh numpy array
    """
    dis = cos_dis(X)
    dis = torch.Tensor(dis)
    _, k_idx = dis.topk(k_neigh, dim=-1, largest=False)
    return k_idx.numpy()


def construct_edge_list_from_knn(Xs, k_neighs):
    """
    construct concatenated edge list from list of features with kNN from multi-modal
    :param Xs: list of features
    :param k_neighs: list of k
    :return: concatenated edge list
    """
    return np.concatenate([_construct_edge_list_from_distance(Xs[i], k_neighs[i]) for i in range(len(Xs))], axis=1)


def _construct_edge_list_from_cluster(X, clusters, adjacent_clusters, k_neighbors) -> np.array:
    """
    construct edge list (numpy array) from cluster for single modality
    :param X: feature
    :param clusters: number of clusters for k-means
    :param adjacent_clusters: a node's adjacent clusters
    :param k_neighbors: number of a node's neighbors
    :return:
    """
    N = X.shape[0]
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    dis = euclidean_distances(X, centers)
    _, cluster_center_dict = torch.topk(torch.Tensor(dis), adjacent_clusters, largest=False)
    cluster_center_dict = cluster_center_dict.numpy()
    point_labels = kmeans.labels_
    point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(clusters)]

    def _list_cat(list_of_array):
        """
        example: [[0,1],[3,5,6],[-1]] -> [0,1,3,5,6,-1]
        :param list_of_array: list of np.array
        :return: list of numbers
        """
        ret = list()
        for array in list_of_array:
            ret += array.tolist()
        return ret

    cluster_neighbor_dict = [_list_cat([point_in_which_cluster[cluster_center_dict[point][i]]
                                        for i in range(adjacent_clusters)]) for point in range(N)]
    for point, entry in enumerate(cluster_neighbor_dict):
        entry.append(point)
    sampled_ids = [SampledGraphConvolution.sample_ids(cluster_neighbor_dict[point], k_neighbors) for point in range(N)]
    return np.array(sampled_ids)


def construct_edge_list_from_cluster(Xs, clusters, adjacent_clusters, k_neighbors) -> np.array:
    """
    construct concatenated edge list from list of features with cluster from multi-modal
    :param Xs: list of features of each modality
    :param clusters: list of number of clusters for k-means of each modality
    :param adjacent_clusters: list of number of a node's adjacent clusters of each modality
    :param k_neighbors: list of number of a node's neighbors
    :return: concatenated edge list (numpy array)
    """
    return np.concatenate([_construct_edge_list_from_cluster(Xs[i], clusters[i], adjacent_clusters[i], k_neighbors[i])
                           for i in range(len(Xs))], axis=1)