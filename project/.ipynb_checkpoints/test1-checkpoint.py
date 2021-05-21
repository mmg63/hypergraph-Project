#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:59:17 2020

@author: mustafamohammadi
"""

#import torch_geometric.nn.conv.hypergraph_conv as hconv
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root="./Cora", name="Cora")


data1 = dataset[0]

print(data1.is_undirected())

print(data1.val_mask.sum().item())
print(data1.test_mask.sum().item)
print()

        




