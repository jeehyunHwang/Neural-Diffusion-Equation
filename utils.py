"""
Code developed based on "https://github.com/sungyongs/dpgn"
"""
import sys
import os.path as osp
from itertools import repeat

import networkx as nx

import torch
from torch_sparse import coalesce
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_scatter import scatter_add
import numpy as np


def get_edge_index_from_nxG(G):
    """return edge_index for torch_geometric.data.data.Data
    G is networkx Graph.
    """
    A = nx.adj_matrix(G)    # A: sparse.csr_matrix
    r, c = A.nonzero()
    r = torch.tensor(r, dtype=torch.long)
    c = torch.tensor(c, dtype=torch.long)
    
    return torch.stack([r,c])


def maybe_num_nodes(edge_index, num_nodes=None):
    return edge_index.max().item() + 1 if num_nodes is None else num_nodes


def remove_self_loops(edge_index, edge_attr=None):
    row, col = edge_index
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    mask = mask.unsqueeze(0).expand_as(edge_index)
    edge_index = edge_index[mask].view(2, -1)

    return edge_index, edge_attr


def add_self_loops(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    dtype, device = edge_index.dtype, edge_index.device
    loop = torch.arange(0, num_nodes, dtype=dtype, device=device)
    loop = loop.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop], dim=1)

    return edge_index


def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    # NOTE: There are duplicated edges and self loops in the datasets. Other
    # implementations do not remove them!
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index


def degree(index, num_nodes=None, dtype=None, device=None):
    """Computes the degree of a given index tensor.
    Args:
        index (LongTensor): Source or target indices of edges.
        num_nodes (int, optional): The number of nodes in :attr:`index`.
            (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional). The desired data type of returned
            tensor.
        device (:obj:`torch.device`, optional): The desired device of returned
            tensor.
    :rtype: :class:`Tensor`
    .. testsetup::
        import torch
    .. testcode::
        from torch_geometric.utils import degree
        index = torch.tensor([0, 1, 0, 2, 0])
        output = degree(index)
        print(output)
    .. testoutput::
       tensor([ 3.,  1.,  1.])
    """
    num_nodes = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((num_nodes), dtype=dtype, device=device)
    return out.scatter_add_(0, index, out.new_ones((index.size(0))))


def normalized_cut(edge_index, edge_attr, num_nodes=None):
    row, col = edge_index
    deg = 1 / degree(row, num_nodes, edge_attr.dtype, edge_attr.device)
    deg = deg[row] + deg[col]
    cut = edge_attr * deg
    return cut


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def get_adj(edge_index, weight=None, augmented=False, undirected=True):
    """return adjacency matrix"""
    if not weight:
        weight = torch.ones(edge_index.shape[1])

    row, col = edge_index
    # if undirected:
    #     adj = np.maximum.reduce([adj.asarray(), adj.T]).to_sparse()
    # else:
    adj = torch.sparse.FloatTensor(edge_index, weight)
    identity = torch.eye(adj.shape[0]).to_sparse()

    if augmented==True:
        return adj+identity
    else:
        return adj

def get_laplacian(edge_index, weight=None, type='norm', sparse=True):
    """return Laplacian (sparse tensor)
    type: 'comb' or 'norm' for combinatorial or normalized one.
    """
    adj = get_adj(edge_index, weight=weight)    # torch.sparse.FloatTensor
    num_nodes = adj.shape[1]
    senders, receivers = edge_index
    num_edges = edge_index.shape[1]
    
    deg = scatter_add(torch.ones(num_edges), senders)
    sp_deg = torch.sparse.FloatTensor(torch.tensor([range(num_nodes),range(num_nodes)]), deg)
    Laplacian = sp_deg - adj    # L = D-A
    
    deg = deg.pow(-0.5)
    deg[deg == float('inf')] = 0
    sp_deg = torch.sparse.FloatTensor(torch.tensor([range(num_nodes),range(num_nodes)]), deg)
    Laplacian_norm = sp_deg.mm(Laplacian.mm(sp_deg.to_dense()))     # Lsym = (D^-1/2)L(D^-1/2)
    
    if type=="comb":
        return Laplacian if sparse else Laplacian.to_dense()
    elif type=="norm":
        return to_sparse(Laplacian_norm) if sparse else Laplacian_norm
    elif type=="aug":
        aug_adj = get_adj(edge_index, weight=weight, augmented=True)
        num_nodes = aug_adj.shape[1]
        senders, receivers = edge_index
        num_edges = edge_index.shape[1]
        
        deg = scatter_add(torch.ones(num_edges), senders)
        sp_deg = torch.sparse.FloatTensor(torch.tensor([range(num_nodes),range(num_nodes)]), deg)
        Laplacian = sp_deg - aug_adj 

        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        sp_deg = torch.sparse.FloatTensor(torch.tensor([range(num_nodes),range(num_nodes)]), deg)
        aug_Laplacian_norm = sp_deg.mm(Laplacian.mm(sp_deg.to_dense())) 
        return to_sparse(aug_Laplacian_norm) if sparse else aug_Laplacian_norm
    else:
        raise ValueError("type should be one of ['comb', 'norm']")

def decompose_graph(graph):
    # graph: torch_geometric.data.data.Data
    x, edge_index, edge_attr, global_attr = None, None, None, None
    for key in graph.keys:
        if key=="x":
            x = graph.x
        elif key=="edge_index":
            edge_index = graph.edge_index
        elif key=="edge_attr":
            edge_attr = graph.edge_attr
        elif key=="global_attr":
            global_attr = graph.global_attr
        else:
            pass
    return (x, edge_index, edge_attr, global_attr)


def graph_concat(graph1, graph2, 
                 node_cat=True, edge_cat=True, global_cat=False):
    """
    Args:
        graph1: torch_geometric.data.data.Data
        graph2: torch_geometric.data.data.Data
        node_cat: True if concat node_attr
        edge_cat: True if concat edge_attr
        global_cat: True if concat global_attr
    Return:
        new graph: concat(graph1, graph2)
    """
    # graph2 attr is used for attr that is not concated.
    _x = graph2.x
    _edge_attr = graph2.edge_attr
    _global_attr = graph2.global_attr
    _edge_index = graph2.edge_index
    
    if node_cat:
        try:
            _x = torch.cat([graph1.x, graph2.x], dim=-1)
        except:
            raise ValueError("Both graph1 and graph2 should have 'x' key.")
    
    if edge_cat:
        try:
            _edge_attr = torch.cat([graph1.edge_attr, graph2.edge_attr], dim=-1)
        except:
            raise ValueError("Both graph1 and graph2 should have 'edge_attr' key.")
            
    if global_cat:
        try:
            _global_attr = torch.cat([graph1.global_attr, graph2.global_attr], dim=-1)
        except:
            raise ValueError("Both graph1 and graph2 should have 'global_attr' key.")

    ret = Data(x=_x, edge_attr=_edge_attr, edge_index=_edge_index)
    ret.global_attr = _global_attr
    
    return ret


def copy_geometric_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    node_attr, edge_index, edge_attr, global_attr = decompose_graph(graph)
    
    ret = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    ret.global_attr = global_attr
    
    return ret

def z_score(x, mean, std, dtype=None, device=None):
    if dtype=='tensor':
        mean = torch.tensor(mean, dtype=torch.float64, device=device)
        std = torch.tensor(std, dtype=torch.float64, device=device)
        return torch.div(torch.subtract(x, mean), std)
    else:
        return (x - mean) / std

def z_inverse(x, mean, std, dtype=None, device=None):
    if dtype=='tensor':
        mean = torch.tensor(mean, dtype=torch.float64, device=device)
        std = torch.tensor(std, dtype=torch.float64, device=device)
        return torch.add(torch.mul(x, std), mean)
    else:
        return (x * std) + mean

def get_ffnn(input_size, output_size, nn_desc, dropout_rate, bias, bn):
    """
    Derived from "https://github.com/HerreraKrachTeichmann/NJODE"
    function to get a feed-forward neural network with the given description
    :param input_size: int, input dimension
    :param output_size: int, output dimension
    :param nn_desc: list of lists or None, each inner list defines one hidden
            layer and has 2 elements: 1. int, the hidden dim, 2. str, the
            activation function that should be applied (see dict nonlinears for
            possible options)
    :param dropout_rate: float,
    :param bias: bool, whether a bias is used in the layers
    :return: torch.nn.Sequential, the NN function
    """
    nonlinears = {
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU
    }
    if nn_desc is None or (len(nn_desc) == 1 and len(nn_desc[0]) == 1):
        layers = [torch.nn.Linear(input_size, output_size, bias=bias)]
        if len(nn_desc) == 1:
            layers.append(nonlinears[nn_desc[0][0]]())
    else:
        layers = [torch.nn.Linear(input_size, nn_desc[0][0], bias=bias)]
        if bn:
            layers.append(torch.nn.BatchNorm1d(nn_desc[0][0]))
        if len(nn_desc) > 1:
            for i in range(len(nn_desc)-2):
                layers.append(nonlinears[nn_desc[i][1]]())
                layers.append(torch.nn.Dropout(p=dropout_rate))
                layers.append(torch.nn.Linear(nn_desc[i][0], nn_desc[i+1][0],
                                              bias=bias))
        layers.append(nonlinears[nn_desc[-2][1]]())
        layers.append(torch.nn.Dropout(p=dropout_rate))
        layers.append(torch.nn.Linear(nn_desc[-2][0], output_size, bias=bias))
        if nn_desc[-1][0] == None:
            return torch.nn.Sequential(*layers)
        else: 
            layers.append(nonlinears[nn_desc[-1][0]]())
    return torch.nn.Sequential(*layers)
