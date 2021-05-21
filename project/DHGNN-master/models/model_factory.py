from models.models import *


def model_select(activate_model):
    model_dict = {'MLP': MLP,
                  'EdgeConvNet': EdgeConvNet,
                  'CentroidEdgeConvNet': CentroidEdgeConvNet,
                  'CentroidUOMNet': CentroidUOMNet,
                  'SampledGCN': SampledGCN,
                  'GCN': GCN,
                  'TopTGCN': TopTGCN,
                  'TransGCN_v0': TransGCN_v0,
                  'TransGCN_v1': TransGCN_v1,
                  'TransGCN_v2': TransGCN_v2,
                  'TransGCN_v3': TransGCN_v3,
                  'TransGCN_v4': TransGCN_v4,
                  'TransGCN_v5': TransGCN_v5,
                  'DynamicGCN': DynamicGCN,
                  'MultiInputMLP': MultiInputMLP,
                  'HGNN': HGNN
                  }
    return model_dict[activate_model]