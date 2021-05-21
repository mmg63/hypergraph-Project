import torch
from torch import nn
from torch.nn import Module
from models.layers import *
import pandas as pd


class MLP(Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        self.layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + self.layer_spec
        self.dims_out = self.layer_spec + [self.n_categories]
        self.fcs = nn.ModuleList([nn.Linear(self.dims_in[i], self.dims_out[i]) for i in range(self.n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(p=kwargs['dropout_rate']) for i in range(self.n_layers)])
        self.activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])

    def forward(self, **kwargs):
        ids = kwargs['ids']
        feats = kwargs['feats']

        x = feats[ids]
        for i_layer in range(self.n_layers):
            x = self.activations[i_layer](self.fcs[i_layer](self.dropouts[i_layer](x)))
        return x


class EdgeConvNet(Module):
    """
    Unsampling version (full adjacent hyperedges and full containing nodes)
    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        # dim_feat,
        # n_categories,
        # n_layers=2,
        # layer_spec=[128]
        # dropout_rate=0.5
        """
        super(EdgeConvNet, self).__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        self.layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + self.layer_spec
        self.dims_out = self.layer_spec + [self.n_categories]
        self.fcs = nn.ModuleList([nn.Linear(self.dims_in[i], self.dims_out[i]) for i in range(self.n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(p=kwargs['dropout_rate']) for i in range(self.n_layers)])
        self.activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers-1)] + [nn.LogSoftmax(dim=-1)])

    def aggregate(self, **kwargs):
        raise NotImplementedError
        # efeats = None
        # for idx in ids:
        #     node_concerned_edges = node_dict[idx]
        #     edge_feats = None
        #     for edge_id in node_concerned_edges:
        #         nodes_in_edge = edge_dict[edge_id]
        #         if i_layer == 0:
        #             node_embs = feats[nodes_in_edge]
        #         else:
        #             node_embs = self.aggregate(nodes_in_edge, feats, node_dict, edge_dict, i_layer - 1)
        #         edge_emb = node_embs.max(dim=0)[0]         # node feats -> edge feat: max pooling
        #         if edge_feats is None:
        #             edge_feats = edge_emb.reshape((1, edge_emb.size()[0]))
        #         else:
        #             edge_feats = torch.cat((edge_feats, edge_emb.reshape((1, edge_emb.size()[0]))), dim=0)
        #     efeat = edge_feats.max(dim=0)[0]                # edge feats -> node feat: max pooling
        #     if efeats is None:
        #         efeats = efeat.reshape((1, efeat.size()[0]))
        #     else:
        #         efeats = torch.cat((efeats, efeat.reshape((1, efeat.size()[0]))), dim=0)
        # if i_layer == 0:
        #     embs = feats[ids]
        # else:
        #     embs = self.aggregate(ids, feats, node_dict, edge_dict, i_layer - 1)
        # try:
        #     aggregated = torch.cat((embs, efeats), dim=1)
        # except TypeError:
        #     print(ids)
        #     print("Empty batch!")
        # return self.activations[i_layer]((self.fcs[i_layer](aggregated)))

    def forward(self, **kwargs):
        raise NotImplementedError


class CentroidEdgeConvNet(EdgeConvNet):
    def __init__(self, **kwargs):
        super(CentroidEdgeConvNet, self).__init__(**kwargs)

    def aggregate(self, **kwargs):
        """
        :param ids:
        :param feats:
        :param edge_dict:
        :param i_layer:
        :return:
        """
        ids = kwargs['ids']
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']
        i_layer = kwargs['i_layer']

        if i_layer == 0:
            return feats[ids]
        else:
            aggregated = None
            for idx in ids:
                nodes_in_edge = edge_dict[idx]      # only sample the centroid hyperedge
                node_embs = self.aggregate(ids=nodes_in_edge, feats=feats, edge_dict=edge_dict, i_layer=i_layer - 1)
                # edge_emb = node_embs.max(dim=0)[0]  # node feats -> edge feat: max pooling
                pooled_emb = node_embs.mean(dim=0)    # node feats -> edge feat: mean pooling
                if aggregated is None:
                    aggregated = pooled_emb.reshape((1, pooled_emb.size()[0]))
                else:
                    aggregated = torch.cat((aggregated, pooled_emb.reshape((1, pooled_emb.size()[0]))), dim=0)
            return self.activations[i_layer-1](self.fcs[i_layer-1](self.dropouts[i_layer-1](aggregated)))

    def forward(self, **kwargs):
        return self.aggregate(ids=kwargs['ids'], feats=kwargs['feats'],
                              edge_dict=kwargs['edge_dict'], i_layer=self.n_layers)


class CentroidUOMNet(CentroidEdgeConvNet):
    """
    Centroid Convolution Net with Sampling and Unordered Mapping
    """
    def __init__(self, **kwargs):
        super(CentroidUOMNet, self).__init__(**kwargs)

        self.k_sample = kwargs['k_sample']

        self.dims_in = [self.dim_feat] + self.layer_spec[:-1]
        self.dims_out = self.layer_spec
        self.conv = nn.ModuleList([nn.Conv1d(self.dims_in[i], self.dims_out[i], 3) for i in range(self.n_layers)])

        self.fc_end = nn.Linear(self.dims_out[-1], self.n_categories)
        self.activation_end = nn.LogSoftmax(dim=-1)

        self.uom_fc = nn.ModuleList([nn.Linear(self.dims_in[i], self.k_sample*self.k_sample) for i in range(self.n_layers)])
        self.uom_activation = nn.ModuleList([nn.ReLU() for i in range(self.n_layers)])

    def aggregate(self, **kwargs):
        """
        :param ids:
        :param feats:
        :param edge_dict:
        :param i_layer:
        :param device:
        :return:
        """
        ids = kwargs['ids']
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']
        i_layer = kwargs['i_layer']
        device = kwargs['device']

        if i_layer == 0:
            return feats[ids]
        else:
            N = len(ids)
            k = self.k_sample
            d = self.dims_in[i_layer - 1]
            aggregated = torch.zeros(N, k, d)           # (N, k, d)
            for j, idx in enumerate(ids):
                nodes_in_edge = edge_dict[idx]
                node_embs = self.aggregate(ids=nodes_in_edge, feats=feats, edge_dict=edge_dict,
                                           i_layer=i_layer - 1, device=device)
                # edge_emb = node_embs.max(dim=0)[0]    # node feats -> edge feat: max pooling
                # pooled_emb = node_embs.mean(dim=0)    # node feats -> edge feat: mean pooling
                pooled_emb = self.unorder_mapping(node_embs, i_layer)
                aggregated[j] = pooled_emb
            aggregated = aggregated.reshape(N, d, k)    # (N, d1, k)
            aggregated = aggregated.to(device)
            conved = self.activations[i_layer-1](self.dropouts[i_layer-1](self.conv[i_layer-1](aggregated)))   # (N, d2, k)
            conved = conved.mean(dim=2)                 # (N, d2)
            return conved

    def unorder_mapping(self, embs, i_layer):
        """
        sample and transfer node embeddings to unordered embeddings
        :param sampled_embs: sampled embeddings from node_embs
        :return: unordered embeddings
        """
        emb_ids = list(range(embs.shape[0]))
        df = pd.DataFrame(emb_ids)
        sampled_ids = df.sample(self.k_sample - 1, replace=True).values
        sampled_ids = sampled_ids.reshape((sampled_ids.size, )).tolist()
        sampled_ids.append(emb_ids[-1])             # must sample the centroid node itself
        sampled_embs = embs[sampled_ids]

        mapping1d = self.uom_activation[i_layer-1](self.uom_fc[i_layer-1](sampled_embs))   # (k, d) -> (k, k*k)
        mapping1d = mapping1d.mean(dim=0)                                       # (k, k*k) -> (1, k*k)
        mapping2d = mapping1d.reshape(self.k_sample, self.k_sample)             # (1, k*k) -> (k, k)
        unordered_embs = torch.matmul(mapping2d, sampled_embs)                  # (k, k) * (k, d) -> (k, d)
        return unordered_embs

    def forward(self, **kwargs):
        """
        :param ids:
        :param feats:
        :param edge_dict:
        :param device:
        :return:
        """
        ids = kwargs['ids']
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']
        device = kwargs['device']

        embs = self.aggregate(ids=ids, feats=feats, edge_dict=edge_dict, i_layer=self.n_layers, device=device)  # (N, d)
        output = self.activation_end((self.fc_end(embs)))
        return output


class GCN(nn.Module):
    """
    GCN implementation with node wise region aggregation
    """
    def __init__(self, **kwargs):
        """
                :param kwargs:
                # dim_feat
                # n_categories
                # n_layers=2
                # layer_spec=[128]
                # dropout_rate=0.5
                # has_bias = False
                """
        super().__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([GraphConvolution(
            dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[i],
            has_bias=kwargs['has_bias']) for i in range(self.n_layers)])

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']

        x = feats                                   # (N, d)
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](x, edge_dict)   # (N, d)
        return x


class SampledGCN(nn.Module):
    """
    Whole graph based convolution with unordered mapping
    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        # dim_feat,
        # n_categories,
        # n_layers=2,
        # layer_spec=[128]
        # k_sample = 100
        # dropout_rate=0.5
        """
        super().__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([SampledGraphConvolution(
            dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[i],
            k_sample=kwargs['k_sample'],
            has_bias=kwargs['has_bias']) for i in range(self.n_layers)])

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']

        x = feats
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](x, edge_dict)
        return x


class TransGCN_v0(nn.Module):
    """
    Two TransGraphConvolution network
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for _ in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([TransGraphConvolution(
            dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[i],
            k_sample=kwargs['k_sample'],
            has_bias=kwargs['has_bias']) for i in range(self.n_layers)])

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']

        x = feats
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](x, edge_dict)
        return x


class TransGCN_v1(nn.Module):
    """
    One layer GCN + one layer TransGCN
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([GraphConvolution(
            dim_in=self.dims_in[0],
            dim_out=self.dims_out[0],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[0],
            has_bias=kwargs['has_bias']
            )]
            + [TransGraphConvolution(
            dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[i],
            k_sample=kwargs['k_sample'],
            has_bias=kwargs['has_bias']) for i in range(1, self.n_layers)])

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']

        x = feats
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](x, edge_dict)
        return x


class TopTGCN(nn.Module):
    """
    One layer GCN + one layer TopTGraphConvolution
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        self.t_top = kwargs['t_top']
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([GraphConvolution(
            dim_in=self.dims_in[0],
            dim_out=self.dims_out[0],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[0],
            has_bias=kwargs['has_bias']
            )]
            + [TopTGraphConvolution(
            dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[i],
            k_sample=kwargs['k_sample'],
            t_top=kwargs['t_top'],
            has_bias=kwargs['has_bias']) for i in range(1, self.n_layers)])

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']

        x = feats
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](x, edge_dict)
        return x


class TransGCN_v2(nn.Module):
    """
    One layer GCN + one layer MixedNearestConvolution
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([GraphConvolution(
            dim_in=self.dims_in[0],
            dim_out=self.dims_out[0],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[0],
            has_bias=kwargs['has_bias'])]
            + [MixedNearestConvolution(
            dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[i],
            structured_neighbor=kwargs['k_structured'],
            nearest_neighbor=kwargs['k_nearest'],
            has_bias=kwargs['has_bias']) for i in range(1, self.n_layers)])
    
    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']

        x = feats
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](x, edge_dict)
        return x


class TransGCN_v3(nn.Module):
    """
    One layer GCN + one layer ClusterConvolution
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([GraphConvolution(
            dim_in=self.dims_in[0],
            dim_out=self.dims_out[0],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[0],
            has_bias=kwargs['has_bias'])]
            + [ClusterConvolution(
            dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[i],
            structured_neighbor=kwargs['k_structured'],
            cluster_neighbor=kwargs['k_cluster'],
            n_cluster=kwargs['clusters'],
            n_center=kwargs['adjacent_centers'],
            has_bias=kwargs['has_bias']) for i in range(1, self.n_layers)])

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']

        x = feats
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](x, edge_dict)
        return x


class TransGCN_v4(nn.Module):
    """
    One layer GCN + one layer MultiClusterConvolution
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([GraphConvolution(
            dim_in=self.dims_in[0],
            dim_out=self.dims_out[0],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[0],
            has_bias=kwargs['has_bias'])]
            + [MultiClusterConvolution(
            dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[i],
            structured_neighbor=kwargs['k_structured'],
            nearest_neighbor=kwargs['k_nearest'],
            cluster_neighbor=kwargs['k_cluster'],
            n_cluster=kwargs['clusters'],
            n_center=kwargs['adjacent_centers'],
            has_bias=kwargs['has_bias']) for i in range(1, self.n_layers)])

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']

        x = feats
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](x, edge_dict)
        return x


class TransGCN_v5(nn.Module):
    """
    One layer GCN + one layer MultiClusterConvolution
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([GraphConvolution(
            dim_in=self.dims_in[0],
            dim_out=self.dims_out[0],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[0],
            has_bias=kwargs['has_bias'])]
            + [MultiClusterConvolution_v2(
            dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[i],
            structured_neighbor=kwargs['k_structured'],
            nearest_neighbor=kwargs['k_nearest'],
            cluster_neighbor=kwargs['k_cluster'],
            n_cluster=kwargs['clusters'],
            n_center=kwargs['adjacent_centers'],
            has_bias=kwargs['has_bias']) for i in range(1, self.n_layers)])

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']

        x = feats
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](x, edge_dict)
        return x


class DynamicGCN(nn.Module):
    """
    One layer GCN + one layer MultiClusterConvolution_unconv
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.dim_feat = kwargs['dim_feat']
        self.n_categories = kwargs['n_categories']
        self.n_layers = kwargs['n_layers']
        layer_spec = kwargs['layer_spec']
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([GraphConvolution(
            dim_in=self.dims_in[0],
            dim_out=self.dims_out[0],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[0],
            has_bias=kwargs['has_bias'])]
            + [DynamicMultiCluster(
            dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[i],
            structured_neighbor=kwargs['k_structured'],
            nearest_neighbor=kwargs['k_nearest'],
            cluster_neighbor=kwargs['k_cluster'],
            n_cluster=kwargs['clusters'],
            n_center=kwargs['adjacent_centers'],
            has_bias=kwargs['has_bias']) for i in range(1, self.n_layers)])

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        feats = kwargs['feats']
        edge_dict = kwargs['edge_dict']

        x = feats
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](x, edge_dict)
        return x


class MultiInputMLP(nn.Module):
    """
    two layer multi-input MLP
    feature concatenate after the input layer
    """
    def __init__(self, **kwargs):
        """
        :param n_input -> int: # of inputs
        :param dims_in -> list: list of input dimensions
        :param n_category -> int: # of categories
        :param hiddens -> list: list of hidden layer dimensions for each input
        """
        super().__init__()

        self.n_input = kwargs['n_input']
        self.dims_in = kwargs['dims_in']
        self.n_category = kwargs['n_category']
        self.hiddens = kwargs['hiddens']
        self.input_fcs = nn.ModuleList([nn.Linear(self.dims_in[i], self.hiddens[i])
                                        for i in range(self.n_input)])
        self.input_acts = nn.ModuleList([nn.ReLU() for i in range(self.n_input)])
        self.output_fc = nn.Linear(sum(self.hiddens), self.n_category)
        self.output_act = nn.LogSoftmax(dim=-1)

    def forward(self, fts):
        """
        :param fts: list of input features
        :return: out: model output, temps: input layer output, list of embeddings
        """
        tmps = [self.input_acts[i](self.input_fcs[i](fts[i])) for i in range(self.n_input)]
        tmp = torch.cat(tmps, dim=1)
        out = self.output_act(self.output_fc(tmp))
        return out, tmps


class MultiInputGCN(nn.Module):
    """
    Several MLP layers for multi-input
    + one layer TransGraphConvolution
    (for knn hypergraph)
    """
    def __init__(self, **kwargs):
        """
        :param n_input -> int: # of inputs
        :param dims_in -> list: list of input dimensions
        :param n_category -> int: # of categories
        :param hiddens -> list: list of hidden layer dimensions for each input
        """
        super().__init__()

        self.n_input = kwargs['n_input']
        self.dims_in = kwargs['dims_in']
        self.n_category = kwargs['n_category']
        self.knn = kwargs['knn']
        self.hiddens = kwargs['hiddens']
        self.drop_out = kwargs['drop_out']

        self.input_dropouts = nn.ModuleList([nn.Dropout(self.drop_out) for i in range(self.n_input)])
        self.input_acts = nn.ModuleList([nn.ReLU() for i in range(self.n_input)])
        self.input_fcs = nn.ModuleList([nn.Linear(self.dims_in[i], self.hiddens[i])
                                        for i in range(self.n_input)])
        output_act = nn.LogSoftmax(dim=-1)
        self.output_gc = TransGraphConvolution_unsample\
            (dim_in=sum(self.hiddens),
            dim_out=self.n_category,
            knn=self.knn,
            has_bias=True,
            dropout_rate=self.drop_out,
            activation=output_act)

    def forward(self, fts, edge_dict):
        """
        :param fts: list of input features
        :param edge_dict: concatenated edge list
        :return:
        """
        tmps = [self.input_acts[i](self.input_fcs[i](self.input_dropouts[i](fts[i])))
                                   for i in range(self.n_input)]
        tmp = torch.cat(tmps, dim=1)
        out = self.output_gc(tmp, edge_dict)
        return out


class MultiInputGCN_v2(nn.Module):
    """
    Several MLP layers for multi-input
    + one layer TransGraphConvolution
    """
    def __init__(self, **kwargs):
        """
        :param n_input -> int: # of inputs
        :param dims_in -> list: list of input dimensions
        :param n_category -> int: # of categories
        :param hiddens -> list: list of hidden layer dimensions for each input
        """
        super().__init__()

        self.n_input = kwargs['n_input']
        self.dims_in = kwargs['dims_in']
        self.n_category = kwargs['n_category']
        self.knn = kwargs['knn']
        self.hiddens = kwargs['hiddens']
        self.drop_out = kwargs['drop_out']

        self.input_dropouts = nn.ModuleList([nn.Dropout(self.drop_out) for i in range(self.n_input)])
        self.input_acts = nn.ModuleList([nn.ReLU() for i in range(self.n_input)])
        self.input_fcs = nn.ModuleList([nn.Linear(self.dims_in[i], self.hiddens[i])
                                        for i in range(self.n_input)])
        output_act = nn.LogSoftmax(dim=-1)
        self.output_gc = TransGraphConvolution_unsample\
            (dim_in=sum(self.hiddens),
            dim_out=self.n_category,
            knn=self.knn,
            has_bias=True,
            dropout_rate=self.drop_out,
            activation=output_act)

    def forward(self, fts, edge_dict):
        """
        :param fts: list of input features
        :param edge_dict: concatenated edge list
        :return:
        """
        tmps = [self.input_acts[i](self.input_fcs[i](self.input_dropouts[i](fts[i])))
                                   for i in range(self.n_input)]
        tmp = torch.cat(tmps, dim=1)
        out = self.output_gc(tmp, edge_dict)
        return out



class HGNN(nn.Module):
    def __init__(self, **kwargs):
        """
        :param in_ch:
        :param n_class:
        :param n_hid:
        :param dropout:
        """
        super(HGNN, self).__init__()
        in_ch = kwargs['dim_in']
        n_class = kwargs['n_category']
        n_hid = kwargs['hiddens']
        self.dropout = kwargs['drop_out']
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = nn.functional.relu(self.hgc1(x, G))
        x = nn.functional.dropout(x, self.dropout)
        x = nn.functional.log_softmax(self.hgc2(x, G), dim=-1)
        return x
