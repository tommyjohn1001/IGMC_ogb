from __future__ import print_function

import argparse
import itertools
import math
import multiprocessing as mp
import os
import pdb
import random
import sys
import time
import warnings
from copy import deepcopy

import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import torch
import torch.nn as nn
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import degree
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm

warnings.simplefilter("ignore", ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return ssp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return ssp.csc_matrix((data, indices, indptr), shape=shape)


class MyDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        A,
        links,
        labels,
        h,
        sample_ratio,
        max_nodes_per_hop,
        u_features,
        v_features,
        class_values,
        pe_dim,
        metric,
        max_num=None,
        parallel=True,
    ):
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.u_features = u_features
        self.v_features = v_features
        self.class_values = class_values
        self.parallel = parallel
        self.max_num = max_num
        self.pe_dim = pe_dim
        self.metric = metric

        if max_num is not None:
            np.random.seed(123)
            num_links = len(links[0])
            perm = np.random.permutation(num_links)
            perm = perm[:max_num]
            self.links = (links[0][perm], links[1][perm])
            self.labels = labels[perm]
        super(MyDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        name = "data.pt"
        if self.max_num is not None:
            name = "data_{}.pt".format(self.max_num)
        return [name]

    def process(self):
        # Extract enclosing subgraphs and save to disk
        data_list = links2subgraphs(
            self.Arow,
            self.Acol,
            self.links,
            self.labels,
            self.h,
            self.pe_dim,
            self.metric,
            self.sample_ratio,
            self.max_nodes_per_hop,
            self.u_features,
            self.v_features,
            self.class_values,
            self.parallel,
        )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        del data_list


class MyDynamicDataset(Dataset):
    def __init__(
        self,
        root,
        A,
        links,
        labels,
        h,
        sample_ratio,
        max_nodes_per_hop,
        u_features,
        v_features,
        class_values,
        pe_dim,
        max_num=None,
    ):
        super(MyDynamicDataset, self).__init__(root)
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.u_features = u_features
        self.v_features = v_features
        self.class_values = class_values
        self.pe_dim = pe_dim
        if max_num is not None:
            np.random.seed(123)
            num_links = len(links[0])
            perm = np.random.permutation(num_links)
            perm = perm[:max_num]
            self.links = (links[0][perm], links[1][perm])
            self.labels = labels[perm]

    def __len__(self):
        return len(self.links[0])

    def get(self, idx):
        i, j = self.links[0][idx], self.links[1][idx]
        g_label = self.labels[idx]
        tmp = subgraph_extraction_labeling(
            (i, j),
            self.Arow,
            self.Acol,
            self.h,
            self.sample_ratio,
            self.max_nodes_per_hop,
            self.u_features,
            self.v_features,
            self.class_values,
            g_label,
        )
        return construct_pyg_graph(*tmp, self.pe_dim)


def links2subgraphs(
    Arow,
    Acol,
    links,
    labels,
    h=1,
    pe_dim=1,
    metric="L1",
    sample_ratio=1.0,
    max_nodes_per_hop=None,
    u_features=None,
    v_features=None,
    class_values=None,
    parallel=True,
):
    # extract enclosing subgraphs
    print("Enclosing subgraph extraction begins...")
    g_list = []
    parallel = True

    if not parallel:
        with tqdm(total=len(links[0])) as pbar:
            for i, j, g_label in zip(links[0], links[1], labels):
                tmp = subgraph_extraction_labeling(
                    (i, j),
                    Arow,
                    Acol,
                    h,
                    sample_ratio,
                    max_nodes_per_hop,
                    u_features,
                    v_features,
                    class_values,
                    g_label,
                )
                data = construct_pyg_graph(*tmp, pe_dim)
                if data is not None:
                    g_list.append(data)
                pbar.update(1)
    else:
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap_async(
            subgraph_extraction_labeling,
            [
                (
                    (i, j),
                    Arow,
                    Acol,
                    h,
                    sample_ratio,
                    max_nodes_per_hop,
                    u_features,
                    v_features,
                    class_values,
                    g_label,
                )
                for i, j, g_label in zip(links[0], links[1], labels)
            ],
        )
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready():
                break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
        pbar.close()
        end = time.time()
        print("Time elapsed for subgraph extraction: {}s".format(end - start))
        print("Transforming to pytorch_geometric graphs...")
        g_list = []
        pbar = tqdm(total=len(results))
        while results:
            tmp = results.pop()
            data = construct_pyg_graph(*tmp, pe_dim, metric)
            if data is not None:
                g_list.append(data)
            pbar.update(1)
        pbar.close()
        end2 = time.time()
        print("Time elapsed for transforming to pytorch_geometric graphs: {}s".format(end2 - end))
    return g_list


def subgraph_extraction_labeling(
    ind,
    Arow,
    Acol,
    h=1,
    sample_ratio=1.0,
    max_nodes_per_hop=None,
    u_features=None,
    v_features=None,
    class_values=None,
    y=1,
):
    # extract the h-hop enclosing subgraph around link 'ind'
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])
    for dist in range(1, h + 1):
        v_fringe, u_fringe = neighbors(u_fringe, Arow), neighbors(v_fringe, Acol)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)
        if sample_ratio < 1.0:
            u_fringe = random.sample(u_fringe, int(sample_ratio * len(u_fringe)))
            v_fringe = random.sample(v_fringe, int(sample_ratio * len(v_fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                u_fringe = random.sample(u_fringe, max_nodes_per_hop)
            if max_nodes_per_hop < len(v_fringe):
                v_fringe = random.sample(v_fringe, max_nodes_per_hop)
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)
    subgraph = Arow[u_nodes][:, v_nodes]
    # remove link between target nodes
    subgraph[0, 0] = 0

    # prepare pyg graph constructor input
    u, v, r = ssp.find(subgraph)  # r is 1, 2... (rating labels + 1)
    v += len(u_nodes)
    r = r - 1  # transform r back to rating label
    num_nodes = len(u_nodes) + len(v_nodes)
    node_labels = [x * 2 for x in u_dist] + [x * 2 + 1 for x in v_dist]
    node_index = u_nodes + v_nodes
    max_node_label = 2 * h + 1
    y = class_values[y]

    # get node features
    if u_features is not None:
        u_features = u_features[u_nodes]
    if v_features is not None:
        v_features = v_features[v_nodes]
    node_features = None

    # only output node features for the target user and item
    if u_features is not None and v_features is not None:
        node_features = [u_features[0], v_features[0]]

    return u, v, r, node_labels, max_node_label, y, node_features, node_index


def init_positional_encoding(g, pos_enc_dim):
    """
    Initializing positional encoding with RWPE
    """

    if len(g.edge_type) == 0:
        node_feat = g.x
        PE = torch.zeros(
            (node_feat.size(0), pos_enc_dim), dtype=node_feat.dtype, device=node_feat.device
        )
    else:
        # Geometric diffusion features with Random Walk
        A = ssp.csr_matrix(pyg_utils.to_dense_adj(g.edge_index).squeeze().numpy())
        Dinv = ssp.diags(pyg_utils.degree(g.edge_index[0]).numpy().clip(1) ** -1.0)  # D^-1
        RW = A * Dinv
        M = RW

        # Iterate
        nb_pos_enc = pos_enc_dim
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc - 1):
            M_power = M_power * M
            PE.append(torch.from_numpy(M_power.diagonal()).float())

        PE = torch.stack(PE, dim=-1)

    ## Concate PE to node feat
    g.x = torch.cat((g.x, PE), -1)


def get_features_sp_sample(g, max_sp):
    max_sp -= 2

    G = to_networkx(g)
    target_user = torch.where(g.x[:, 1] == 1)[0].item()
    target_item = torch.where(g.x[:, 2] == 1)[0].item()
    target_node_set = np.array([target_user, target_item])

    dim = max_sp + 2
    set_size = len(target_node_set)
    sp_length = np.ones((G.number_of_nodes(), set_size), dtype=np.int32) * -1
    for i, node in enumerate(target_node_set):
        for node_ngh, length in nx.shortest_path_length(G, source=node).items():
            sp_length[node_ngh, i] = length
    sp_length = np.minimum(sp_length, max_sp)
    onehot_encoding = np.eye(dim, dtype=np.float64)  # [n_features, n_features]
    features_sp = onehot_encoding[sp_length].sum(axis=1)

    ## Concate PE to node feat
    g.x = torch.cat((g.x, torch.from_numpy(features_sp)), -1).float()


def get_non_edges(example: Data, k: int = 50) -> list:
    user_node_idx = torch.where((example.x[:, 1] == 1) | (example.x[:, 3] == 1))
    item_node_idx = torch.where((example.x[:, 2] == 1) | (example.x[:, 4] == 1))
    users = set(user_node_idx[0].tolist())
    items = set(item_node_idx[0].tolist())

    edge_index = example.edge_index.permute((1, 0))
    edges = set(tuple(x) for x in edge_index.tolist())
    possible_edges = set(itertools.product(users, items))

    non_edges = possible_edges.difference(edges)
    if non_edges == []:
        return []

    ## Take k edges only
    k = min(k, len(non_edges))
    non_edges = random.sample(non_edges, k=k)

    return non_edges


def get_rwpe(A, D, pos_enc_dim=5):

    # Geometric diffusion features with Random Walk
    A = ssp.csr_matrix(A.numpy())
    Dinv = ssp.diags(D.numpy().clip(1) ** -1.0)  # D^-1
    RW = A * Dinv
    M = RW

    # Iterate
    nb_pos_enc = pos_enc_dim
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(nb_pos_enc - 1):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal()).float())

    PE = torch.stack(PE, dim=-1)

    return PE


def create_permute_matrix(size):
    indices = list(range(size))
    random.shuffle(indices)

    permutation_map = {old: new for old, new in zip(range(size), indices)}

    permutation_matrix = torch.zeros((size, size))
    for i, j in enumerate(indices):
        permutation_matrix[i][j] = 1

    return permutation_matrix, permutation_map


def create_trg_regu_matrix(trg_user_idx, trg_item_idx, n_nodes: int, metric: str = "L1"):
    """Create target matrix for training regularization loss which is achieved by ContrastiveLoss"""

    trg_matrix = torch.ones((n_nodes, n_nodes))
    S = set([trg_user_idx, trg_item_idx])

    for i in range(n_nodes):
        for j in range(n_nodes):
            condition1 = i not in S and j in S
            condition2 = i in S and j not in S
            if condition1 or condition2:
                trg_matrix[i, j] = 0

    if metric in ["L1", "L2"]:
        trg_matrix = 1 - trg_matrix

    return trg_matrix


def create_permuted_graphs(data: Data, n=10, pos_enc_dim=5, metric: str = "L1") -> list:
    if len(data.edge_index[0]) == 0:
        x_perms = torch.zeros((n, data.x.size(0) + pos_enc_dim))
        targets_perms = torch.zeros((n, 2))

        return x_perms, targets_perms

    ## Get matrix A and D
    D = pyg_utils.degree(data.edge_index[0])
    A = pyg_utils.to_dense_adj(data.edge_index).squeeze(0)

    ## Get target user and item index
    trg_user_idx, trg_item_idx = (
        torch.where(data.x[:, 0] == 1)[0].item(),
        torch.where(data.x[:, 1] == 1)[0].item(),
    )

    permuted_graphs = []
    for _ in range(n):
        perm_matrix, perm_map = create_permute_matrix(len(D))

        new_A = perm_matrix @ A @ perm_matrix.T
        new_D = perm_matrix @ D
        x = get_rwpe(new_A, new_D, pos_enc_dim)

        new_user_idx, new_item_idx = perm_map[trg_user_idx], perm_map[trg_item_idx]
        trg = create_trg_regu_matrix(new_user_idx, new_item_idx, len(D), metric)

        permuted_graphs.append((x.numpy(), trg.numpy()))

    return permuted_graphs


def construct_pyg_graph(
    u, v, r, node_labels, max_node_label, y, node_features, node_index, pos_enc_dim, metric
):
    # TODO: Append node index of each node in u, v, node index is fetched from u_nodes, v_nodes to append to node_feat
    if len(u) == 0:
        return None

    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    node_index = torch.LongTensor(node_index)
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
    edge_type = torch.cat([r, r])
    x = torch.FloatTensor(one_hot(node_labels, max_node_label + 1))
    y = torch.FloatTensor([y])
    data = Data(x, edge_index, edge_type=edge_type, y=y)

    ## Concate with node index
    # NOTE: If using global node index, enable the following
    # data.x = torch.cat([node_index.unsqueeze(-1), data.x], 1)

    if node_features is not None:
        if type(node_features) == list:  # a list of u_feature and v_feature
            u_feature, v_feature = node_features
            data.u_feature = torch.FloatTensor(u_feature).unsqueeze(0)
            data.v_feature = torch.FloatTensor(v_feature).unsqueeze(0)
        else:
            x2 = torch.FloatTensor(node_features)
            data.x = torch.cat([data.x, x2], 1)

    ## Add PE info
    init_positional_encoding(data, pos_enc_dim=pos_enc_dim)

    # NOTE: if using DE, use the following
    ## Add DE info
    # get_features_sp_sample(data, pos_enc_dim)

    ## Add non-edges
    # NOTE: if using idea EdgeAugment, enable the following
    # non_edges = get_non_edges(data)
    # non_edges = torch.tensor(non_edges, dtype=torch.long).permute((1,0))
    # data.non_edge_index = non_edges

    permuted_graphs = create_permuted_graphs(data, n=5, pos_enc_dim=pos_enc_dim, metric=metric)
    data.permuted_graphs = permuted_graphs

    return data


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    if not fringe:
        return set([])
    return set(A[list(fringe)].indices)


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x


def PyGGraph_to_nx(data):
    edges = list(zip(data.edge_index[0, :].tolist(), data.edge_index[1, :].tolist()))
    g = nx.from_edgelist(edges)
    g.add_nodes_from(range(len(data.x)))  # in case some nodes are isolated
    # transform r back to rating label
    edge_types = {(u, v): data.edge_type[i].item() for i, (u, v) in enumerate(edges)}
    nx.set_edge_attributes(g, name="type", values=edge_types)
    node_types = dict(zip(range(data.num_nodes), torch.argmax(data.x, 1).tolist()))
    nx.set_node_attributes(g, name="type", values=node_types)
    g.graph["rating"] = data.y.item()
    return g


class GraphSizeNorm(nn.Module):
    r"""Applies Graph Size Normalization over each individual graph in a batch
    of node features as described in the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x}_i}{\sqrt{|\mathcal{V}|}}
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, batch=None):
        """"""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        inv_sqrt_deg = degree(batch, dtype=x.dtype).pow(-0.5)
        return x * inv_sqrt_deg.index_select(0, batch).view(-1, 1)
