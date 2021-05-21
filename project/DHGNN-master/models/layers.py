import math
import copy
import torch
from torch import nn
from torch.nn.parameter import Parameter
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


class Transform(nn.Module):
    """
    Permutation invariant transformation: (N, k, d) -> (N, k, d)
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()

        if (k * k) % dim_in == 0:
            self.convKK = nn.Conv1d(dim_in, k * k, k, groups=dim_in)
        elif dim_in % (k * k) == 0:
            self.convKK = nn.Conv1d(dim_in, k * k, k, groups=k * k)
        else:
            self.convKK = nn.Conv1d(dim_in, k * k, k)
        # (N, d, k) -> (N, k*k, 1)
        # groups: reduce parameters
        self.activation = nn.Softmax(dim=-1)

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, k, d)
        """
        N, k, d = region_feats.size()  # (N, k, d)
        region_feats = region_feats.permute(0, 2, 1)  # (N, d, k)
        conved = self.convKK(region_feats)  # (N, k*k, 1)
        multiplier = conved.view(N, k, k)  # (N, k, k)
        multiplier = self.activation(multiplier)  # softmax along last dimension
        region_feats = region_feats.permute(0, 2, 1)  # (N, k, d)
        transformed_feats = torch.matmul(multiplier, region_feats)  # (N, k, d)
        return transformed_feats


class ConvMapping(nn.Module):
    """
    Transform (N, k, d) feature to (N, d) feature by transform matrix and 1-D convolution
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()

        self.trans = Transform(dim_in, k)                   # (N, k, d) -> (N, k, d)
        self.convK1 = nn.Conv1d(k, 1, 1)                    # (N, k, d) -> (N, 1, d)

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, d)
        """
        transformed_feats = self.trans(region_feats)
        pooled_feats = self.convK1(transformed_feats)             # (N, 1, d)
        pooled_feats = pooled_feats.squeeze()
        return pooled_feats


class GraphConvolution(nn.Module):
    """
    A GCN layer
    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        # dim_in,
        # dim_out,
        # dropout_rate=0.5,
        # activation
        """
        super().__init__()

        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=kwargs['has_bias'])
        self.dropout = nn.Dropout(p=kwargs['dropout_rate'])
        self.activation = kwargs['activation']

    @staticmethod
    def _region_aggregate(feats, edge_dict):
        N = feats.size()[0]
        pooled_feats = torch.stack([torch.mean(feats[edge_dict[i]], dim=0) for i in range(N)])
        return pooled_feats

    def forward(self, feats, edge_dict):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        x = feats  # (N, d)
        x = self._region_aggregate(x, edge_dict)  # (N, d)
        x = self.activation(self.fc(self.dropout(x)))  # (N, d')
        return x


class SampledGraphConvolution(GraphConvolution):
    """
    A GCN layer with sampling
    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        # dim_in,
        # dim_out,
        # dropout_rate=0.5,
        # activation,
        # k_sample
        """
        super().__init__(**kwargs)

        self.k_sample = kwargs['k_sample']

    def _region_select(self, feats, edge_dict):
        N = feats.size()[0]
        region_feats = torch.stack([feats[SampledGraphConvolution.sample_ids(edge_dict[i], self.k_sample)] for i in range(N)], dim=0)    # (N, k, d)
        return region_feats

    @staticmethod
    def sample_ids(ids, k):
        """
        sample #self.k_sample indexes from ids, must sample the centroid node itself
        :param ids: indexes sampled from
        :param k: number of samples
        :return: sampled indexes
        """
        df = pd.DataFrame(ids)
        sampled_ids = df.sample(k - 1, replace=True).values
        sampled_ids = sampled_ids.flatten().tolist()
        sampled_ids.append(ids[-1])  # must sample the centroid node itself
        return sampled_ids

    @staticmethod
    def sample_ids_v2(ids, k):
        """
        purely sample #self.k_sample indexes from ids
        :param ids: indexes sampled from
        :param k: number of samples
        :return: sampled indexes
        """
        df = pd.DataFrame(ids)
        sampled_ids = df.sample(k, replace=True).values
        sampled_ids = sampled_ids.flatten().tolist()
        return sampled_ids

    def forward(self, feats, edge_dict):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        x = feats                                   # (N, d)
        x = self._region_select(x, edge_dict)       # (N, k, d)
        x = torch.mean(x, dim=1)                    # (N, d)
        x = self.activation(self.fc(self.dropout(x)))     # (N, d')
        return x


class TransGraphConvolution(SampledGraphConvolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.trans = ConvMapping(self.dim_in, self.k_sample)

    def forward(self, feats, edge_dict):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        x = feats                                           # (N, d)
        x = self._region_select(x, edge_dict)               # (N, k, d)
        x = self.trans(x)                                   # (N, d)
        x = self.activation(self.fc(self.dropout(x)))       # (N, d')
        return x


class TransGraphConvolution_unsample(GraphConvolution):
    """
    Fixed size edge_dict, not sampled version of TransGraphConvolution
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.knn = kwargs['knn']

        self.trans = ConvMapping(self.dim_in, self.knn)

    def _region_select(self, feats, edge_dict):
        N = feats.size()[0]
        region_feats = torch.stack([feats[edge_dict[i]] for i in range(N)], dim=0)  # (N, k, d)
        return region_feats

    def forward(self, feats, edge_dict):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        x = feats                                           # (N, d)
        x = self._region_select(x, edge_dict)               # (N, k, d)
        x = self.trans(x)                                   # (N, d)
        x = self.activation(self.fc(self.dropout(x)))       # (N, d')
        return x


class TopTGraphConvolution(SampledGraphConvolution):
    """
    Add t_top selection
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.t_top = kwargs['t_top']
        # self.fc = nn.Linear(self.dim_in * self.t_top, self.dim_out, bias=kwargs['has_bias'])

    def forward(self, feats, edge_dict):
        x = feats                                   # (N, d1)
        x = self._region_select(x, edge_dict)       # (N, k, d1)
        x, _ = torch.topk(x, self.t_top, dim=1)     # (N, t, d1)
        x = torch.mean(x, dim=1)
        # x = x.view(N, self.t_top * d)               # (N, t*d1)
        x = self.activation(self.fc(self.dropout(x)))   # (N, d2)
        return x


class MixedNearestConvolution(GraphConvolution):
    """
    Use both neighbors on graph structures and neighbors of nearest distance on embedding space
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ks = kwargs['structured_neighbor']
        self.kn = kwargs['nearest_neighbor']

        self.trans = ConvMapping(self.dim_in, self.ks + self.kn)

    def _region_select(self, feats, edge_dict):
        N = feats.size()[0]
        region_feats = torch.stack([feats[SampledGraphConvolution.sample_ids(edge_dict[i], self.ks)] for i in range(N)], dim=0)    # (N, k, d)
        return region_feats

    def _nearest_select(self, feats):
        N = feats.size()[0]
        dis = MixedNearestConvolution.cos_dis(feats)
        _, idx = torch.topk(dis, self.kn, dim=1)
        k_nearest = torch.stack([feats[idx[i]] for i in range(N)], dim=0)
        return k_nearest

    @staticmethod
    def cos_dis(X):
        """
        cosine distance
        :param X: (N, d)
        :return: (N, N)
        """
        X = nn.functional.normalize(X)
        XT = X.transpose(0, 1)
        return torch.matmul(X, XT)

    def forward(self, feats, edge_dict):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        x = feats                                           # (N, d)
        x1 = self._region_select(x, edge_dict)              # (N, ks, d)
        x2 = self._nearest_select(x)                        # (N, kn, d)
        x = self.trans(torch.cat([x1, x2], dim=1))          # (N, d)
        x = self.activation(self.fc(self.dropout(x)))       # (N, d')
        return x


class ClusterConvolution(GraphConvolution):
    """
    Use cluster to construct hypergraph
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ks = kwargs['structured_neighbor']
        self.n_cluster = kwargs['n_cluster']              # number of clusters
        self.n_center = kwargs['n_center']                # a node has #n_center adjacent clusters
        self.kc = kwargs['cluster_neighbor']    # a node has #cluster_neighbor adjacent nodes

        self.trans = ConvMapping(self.dim_in, self.ks + self.kc)

    def _region_select(self, feats, edge_dict):
        N = feats.size()[0]
        region_feats = torch.stack([feats[SampledGraphConvolution.sample_ids(edge_dict[i], self.ks)] for i in range(N)], dim=0)    # (N, k, d)
        return region_feats

    def _cluster_select(self, feats: torch.Tensor):
        """
        compute k-means centers and cluster labels of each node
        :param feats:
        :return:
        """
        np_feats = feats.detach().cpu().numpy()
        N = np_feats.shape[0]
        kmeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(np_feats)
        centers = kmeans.cluster_centers_
        dis = euclidean_distances(np_feats, centers)
        _, cluster_center_dict = torch.topk(torch.Tensor(dis), self.n_center, largest=False)
        cluster_center_dict = cluster_center_dict.numpy()
        point_labels = kmeans.labels_
        point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(self.n_cluster)]

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
                                  for i in range(self.n_center)]) for point in range(N)]
        for point, entry in enumerate(cluster_neighbor_dict):
            entry.append(point)
        sampled_ids = [SampledGraphConvolution.sample_ids(cluster_neighbor_dict[point], self.kc) for point in range(N)]
        cluster_feats = torch.stack([feats[sampled_ids[point]] for point in range(N)], dim=0)
        return cluster_feats

    def forward(self, feats, edge_dict):
        x = feats  # (N, d)
        x1 = self._region_select(x, edge_dict)  # (N, ks, d)
        x2 = self._cluster_select(x)            # (N, kc, d)
        x = self.trans(torch.cat([x1, x2], dim=1))  # (N, d)
        x = self.activation(self.fc(self.dropout(x)))  # (N, d')
        return x


class SelfAttention(nn.Module):
    """
    Adjacent clusters self attention layer
    """
    def __init__(self, dim_ft, hidden):
        """
        :param dim_ft: feature dimension
        :param hidden: number of hidden layer neurons
        """
        super().__init__()
        self.fc_0 = nn.Linear(dim_ft, hidden)
        self.fc_1 = nn.Linear(hidden, 1)
        self.act_0 = nn.ReLU()
        self.act_1 = nn.Softmax(dim=-1)

    def forward(self, ft):
        """
        use self attention coefficient to compute weighted average on dim=-2
        :param ft (N, t, d)
        :return: y (N, d)
        """
        att = self.act_1(self.fc_1(self.act_0(self.fc_0(ft))))      # (N, t, 1)
        return torch.sum(att * ft, dim=-2).squeeze()


class CentroidAttention(nn.Module):
    """
    Use centroid vertex feature to do attention for adjacent cluster feature aggregation
    """
    def __init__(self):
        """
        :param dim_ft: feature dimension
        :param hidden: number of hidden layer neurons
        """
        super().__init__()

        self.activation = nn.Softmax(dim=-1)

    def forward(self, c_ft, ft):
        """
        :param c_ft: adjacent cluster features  (N, t, d)
        :param ft: centroid features            (N, d)
        :return:
        """
        N, t, d = c_ft.size()
        att = torch.stack([torch.stack([torch.dot(c_ft[i,j,:], ft[i]) for j in range(t)], dim=0) for i in range(N)], dim=0)
        att = self.activation(att).view(N, t, 1)     # (N, t, 1)
        return torch.sum(att * c_ft, dim=-2).squeeze()


class MultiClusterConvolution(GraphConvolution):
    """
    neighborhood = graph edges + k-NN + top T k-means clusters
    self attention version
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ks = kwargs['structured_neighbor'] # number of sampled nodes in graph adjacency
        self.n_cluster = kwargs['n_cluster']              # number of clusters
        self.n_center = kwargs['n_center']                # a node has #n_center adjacent clusters
        self.kn = kwargs['nearest_neighbor']    # number of the 'k' in k-NN
        self.kc = kwargs['cluster_neighbor']    # number of sampled nodes in a adjacent k-means cluster

        self.trans_s = ConvMapping(self.dim_in, self.ks)    # structured trans
        self.trans_n = ConvMapping(self.dim_in, self.kn)    # nearest trans
        self.trans_c = nn.ModuleList([ConvMapping(self.dim_in, self.kc) for i in range(self.n_cluster)])  # k-means cluster trans
        self.self_attention = SelfAttention(self.dim_in, hidden=self.dim_in//4)

    def _structure_select(self, feats, edge_dict):
        """
        :param feats:
        :param edge_dict:
        :return: mapped graph neighbors
        """
        N = feats.size(0)
        region_feats = torch.stack([feats[SampledGraphConvolution.sample_ids(edge_dict[i], self.ks)]
                                    for i in range(N)], dim=0) # (N, ks, d)
        return self.trans_s(region_feats)                      # (N, d)

    def _nearest_select(self, feats):
        """
        :param feats:
        :return: mapped nearest neighbors
        """
        N = feats.size(0)
        dis = MixedNearestConvolution.cos_dis(feats)
        _, idx = torch.topk(dis, self.kn, dim=1)
        nearest_feature = torch.stack([feats[idx[i]] for i in range(N)], dim=0)     # (N, kn, d)
        return self.trans_n(nearest_feature)                                        # (N, d)

    def _cluster_select(self, feats: torch.Tensor):
        """
        compute k-means centers and cluster labels of each node
        return top #n_cluster nearest cluster transformed features
        :param feats:
        :return: top #n_cluster nearest cluster mapped features
        """
        np_feats = feats.detach().cpu().numpy()
        N = np_feats.shape[0]
        kmeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(np_feats)
        centers = kmeans.cluster_centers_
        dis = euclidean_distances(np_feats, centers)
        _, cluster_center_dict = torch.topk(torch.Tensor(dis), self.n_center, largest=False)
        cluster_center_dict = cluster_center_dict.numpy()
        point_labels = kmeans.labels_
        point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(self.n_cluster)]

        cluster_feats = torch.stack([torch.stack([feats[SampledGraphConvolution.sample_ids_v2
                        (point_in_which_cluster[cluster_center_dict[point][i]], self.kc)]   # (N, n_cluster, kc, d)
                        for i in range(self.n_center)], dim=0) for point in range(N)], dim=0)
        cluster_feats = torch.stack([self.trans_c[i](cluster_feats[:, i, :, :]) for i in range(self.n_center)], dim=1)
        return cluster_feats                           # (N, n_cluster, d)

    def forward(self, feats, edge_dict):
        xs = self._structure_select(feats, edge_dict).view(feats.size(0), 1, feats.size(1))     # (N, 1, d)
        xn = self._nearest_select(feats).view(feats.size(0), 1, feats.size(1))                  # (N, 1, d)
        if self.n_center == 0:
            x = torch.cat((xs, xn), dim=1)          # (N, 2, d)
        else:
            xc = self._cluster_select(feats)        # (N, n_cluster, d)
            x = torch.cat((xs, xn, xc), dim=1)      # (N, 2 + n_cluster, d)
        x = self.self_attention(x)                  # (N, d)
        x = self.activation(self.fc(self.dropout(x)))
        return x


class DynamicMultiCluster(GraphConvolution):
    """
    neighborhood = graph edges + k-NN + top T k-means clusters
    unconvolutioned version (for comparison experiment)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ks = kwargs['structured_neighbor']  # number of sampled nodes in graph adjacency
        self.n_cluster = kwargs['n_cluster']  # number of clusters
        self.n_center = kwargs['n_center']  # a node has #n_center adjacent clusters
        self.kn = kwargs['nearest_neighbor']  # number of the 'k' in k-NN
        self.kc = kwargs['cluster_neighbor']  # number of sampled nodes in a adjacent k-means cluster

        self.trans_s = ConvMapping(self.dim_in, self.ks)  # structured trans
        self.trans_n = ConvMapping(self.dim_in, self.kn)  # nearest trans
        self.trans_c = nn.ModuleList(
            [ConvMapping(self.dim_in, self.kc) for i in range(self.n_cluster)])  # k-means cluster trans
        self.self_attention = SelfAttention(self.dim_in, hidden=self.dim_in // 4)

    def _structure_select(self, feats, edge_dict):
        """
        :param feats:
        :param edge_dict:
        :return: mapped graph neighbors
        """
        N = feats.size(0)
        region_feats = torch.stack([feats[SampledGraphConvolution.sample_ids(edge_dict[i], self.ks)]
                                    for i in range(N)], dim=0)  # (N, ks, d)
        return region_feats.mean(dim=1)  # (N, d)

    def _nearest_select(self, feats):
        """
        :param feats:
        :return: mapped nearest neighbors
        """
        N = feats.size(0)
        dis = MixedNearestConvolution.cos_dis(feats)
        _, idx = torch.topk(dis, self.kn, dim=1)
        nearest_feature = torch.stack([feats[idx[i]] for i in range(N)], dim=0)  # (N, kn, d)
        return nearest_feature.mean(dim=1)  # (N, d)

    def _cluster_select(self, feats: torch.Tensor):
        """
        compute k-means centers and cluster labels of each node
        return top #n_cluster nearest cluster transformed features
        :param feats:
        :return: top #n_cluster nearest cluster mapped features
        """
        np_feats = feats.detach().cpu().numpy()
        N = np_feats.shape[0]
        kmeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(np_feats)
        centers = kmeans.cluster_centers_
        dis = euclidean_distances(np_feats, centers)
        _, cluster_center_dict = torch.topk(torch.Tensor(dis), self.n_center, largest=False)
        cluster_center_dict = cluster_center_dict.numpy()
        point_labels = kmeans.labels_
        point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(self.n_cluster)]

        cluster_feats = torch.stack([torch.stack([feats[SampledGraphConvolution.sample_ids_v2
        (point_in_which_cluster[cluster_center_dict[point][i]], self.kc)]  # (N, n_cluster, kc, d)
                                                  for i in range(self.n_center)], dim=0) for point in range(N)],
                                    dim=0)
        cluster_feats = torch.stack([cluster_feats[:, i, :, :].mean(dim=1) for i in range(self.n_center)],
                                    dim=1)
        return cluster_feats  # (N, n_cluster, d)

    def forward(self, feats, edge_dict):
        xs = self._structure_select(feats, edge_dict).view(feats.size(0), 1, feats.size(1))  # (N, 1, d)
        xn = self._nearest_select(feats).view(feats.size(0), 1, feats.size(1))  # (N, 1, d)
        if self.n_center == 0:
            x = torch.cat((xs, xn), dim=1)  # (N, 2, d)
        else:
            xc = self._cluster_select(feats)  # (N, n_cluster, d)
            x = torch.cat((xs, xn, xc), dim=1)  # (N, 2 + n_cluster, d)
        x = torch.mean(x, dim=1)  # (N, d)
        x = self.activation(self.fc(self.dropout(x)))
        return x


class MultiClusterConvolution_v2(GraphConvolution):
    """
    neighborhood = graph edges + k-NN + top T k-means clusters
    centroid attention version
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ks = kwargs['structured_neighbor'] # number of sampled nodes in graph adjacency
        self.n_cluster = kwargs['n_cluster']              # number of clusters
        self.n_center = kwargs['n_center']                # a node has #n_center adjacent clusters
        self.kn = kwargs['nearest_neighbor']    # number of the 'k' in k-NN
        self.kc = kwargs['cluster_neighbor']    # number of sampled nodes in a adjacent k-means cluster

        self.trans_s = ConvMapping(self.dim_in, self.ks)    # structured trans
        self.trans_n = ConvMapping(self.dim_in, self.kn)    # nearest trans
        self.trans_c = nn.ModuleList([ConvMapping(self.dim_in, self.kc) for i in range(self.n_cluster)])  # k-means cluster trans
        self.centroid_attention = CentroidAttention()

    def _structure_select(self, feats, edge_dict):
        """
        :param feats:
        :param edge_dict:
        :return: mapped graph neighbors
        """
        N = feats.size(0)
        region_feats = torch.stack([feats[SampledGraphConvolution.sample_ids(edge_dict[i], self.ks)]
                                    for i in range(N)], dim=0) # (N, ks, d)
        return self.trans_s(region_feats)                      # (N, d)

    def _nearest_select(self, feats):
        """
        :param feats:
        :return: mapped nearest neighbors
        """
        N = feats.size(0)
        dis = MixedNearestConvolution.cos_dis(feats)
        _, idx = torch.topk(dis, self.kn, dim=1)
        nearest_feature = torch.stack([feats[idx[i]] for i in range(N)], dim=0)     # (N, kn, d)
        return self.trans_n(nearest_feature)                                        # (N, d)

    def _cluster_select(self, feats: torch.Tensor):
        """
        compute k-means centers and cluster labels of each node
        return top #n_cluster nearest cluster transformed features
        :param feats:
        :return: top #n_cluster nearest cluster mapped features
        """
        np_feats = feats.detach().cpu().numpy()
        N = np_feats.shape[0]
        kmeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(np_feats)
        centers = kmeans.cluster_centers_
        dis = euclidean_distances(np_feats, centers)
        _, cluster_center_dict = torch.topk(torch.Tensor(dis), self.n_center, largest=False)
        cluster_center_dict = cluster_center_dict.numpy()
        point_labels = kmeans.labels_
        point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(self.n_cluster)]

        cluster_feats = torch.stack([torch.stack([feats[SampledGraphConvolution.sample_ids_v2
                        (point_in_which_cluster[cluster_center_dict[point][i]], self.kc)]   # (N, n_cluster, kc, d)
                        for i in range(self.n_center)], dim=0) for point in range(N)], dim=0)
        cluster_feats = torch.stack([self.trans_c[i](cluster_feats[:, i, :, :]) for i in range(self.n_center)], dim=1)
        return cluster_feats                           # (N, n_cluster, d)

    def forward(self, feats, edge_dict):
        xs = self._structure_select(feats, edge_dict).view(feats.size(0), 1, feats.size(1))      # (N, 1, d)
        xn = self._nearest_select(feats).view(feats.size(0), 1, feats.size(1))                # (N, 1, d)
        xc = self._cluster_select(feats)                                                      # (N, n_cluster, d)
        x = torch.cat((xs, xn, xc), dim=1)      # (N, 2 + n_cluster, d)
        x = self.centroid_attention(x, feats)              # (N, d)
        x = self.activation(self.fc(self.dropout(x)))
        return x



class HGNN_conv(nn.Module):
    """
    For HGNN model
    """
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = nn.functional.relu(self.hgc1(x, G))
        x = nn.functional.dropout(x, self.dropout)
        x = nn.functional.relu(self.hgc2(x, G))
        return x


class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x




