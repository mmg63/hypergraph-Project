import torch
from torch import nn
from thop import profile


def model_init(model):
    state_dict = model.state_dict()
    for key in state_dict:
        if 'weight' in key:
            nn.init.xavier_uniform_(state_dict[key])
        elif 'bias' in key:
            state_dict[key] = state_dict[key].zero_()


def count_hgnn(model, x, hidden, G, y):
    pass


def count_mlgcn(model, x, y):
    pass