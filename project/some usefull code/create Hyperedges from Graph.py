#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:58:25 2020

@author: mustafamohammadi
"""


import torch
import numpy as np



graph = [[1.,2.,3.,4.], 
     [15.,6.,17.,8.],
     [9.,3.,1.,2.]]

py_list = np.array(graph)

print("maximum value: ", np.max(py_list))


print(type(py_list))

pt_tensor_from_list = torch.LongTensor(py_list)


#print(type(pt_tensor_from_list))
print(pt_tensor_from_list,"\n\n")


#define incident matrix as long tensor
H = torch.zeros((3,int((np.max(py_list))))).long()


for i in range(3):
    for j in range(4):
        H[i][pt_tensor_from_list[i][j]-1] = 1



print(H)



""":return

temphyperedge = np.ndarray((2708, 2708))
for i in range(2708):
    for j in range(len(data_graph[i])):
        print("temphyperedge [{},{}] is set to 1".format(i,data_graph[i][j]))
        temphyperedge[i, (data_graph[i][j])] = 1
hyperedge = torch.LongTensor(temphyperedge)

        
"""

