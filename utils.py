import os
import numpy as np
import torch

def get_data_set(path_to_dir):
    # input: path to directory containing .npy files
    # output adj matrices and marginals
    list_dir = os.listdir(path_to_dir)
    graphs = []
    marginals = []
    for filename in list_dir:
        path_to_graph = os.path.join(path_to_dir, filename)
        data_dict = np.load(path_to_graph, allow_pickle=True)[()]
        adj = data_dict['W']
        np.fill_diagonal(adj, data_dict['b'])
        graphs.append(adj)
        marginals.append(data_dict['marginal'])
    return torch.tensor(graphs).float(), torch.tensor(marginals).float()

def evaluate_error(pred, true):
    error = np.mean(np.abs(pred[:,1]-true[:,1]))
    return error

def evaluate_map_acc(pred, true, threshold=0.5):
    pred = pred[:,1].copy()
    true = true[:,1].copy()
    pred[pred > threshold] = True
    pred[pred <= threshold] = False
    true[true > threshold] = True
    true[true <= threshold] = False
    acc = 1 - np.mean(np.abs(pred-true))
    return acc