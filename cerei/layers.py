import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch import Tensor
from torch.nn import Identity, Parameter
from torch_scatter import scatter
from torch_sparse import SparseTensor, masked_select_nnz, matmul

from cerei.util_functions import *

Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]


def uniform(size: int, value: Any):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, "parameters") else []:
            uniform(size, v)
        for v in value.buffers() if hasattr(value, "buffers") else []:
            uniform(size, v)


def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, "parameters") else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, "buffers") else []:
            glorot(v)


def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, "parameters") else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, "buffers") else []:
            constant(v, fill_value)


def zeros(value: Any):
    constant(value, 0.0)


def reset(value: Any):
    if hasattr(value, "reset_parameters"):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, "children") else []:
            reset(child)


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (Tensor, Tensor) -> Tensor
    pass


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (SparseTensor, Tensor) -> SparseTensor
    pass


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]

    return masked_select_nnz(edge_index, edge_mask, layout="coo")


class GatedGCNLayer(pyg_nn.MessagePassing):
    """
    GatedGCN layer
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """

    def __init__(self, in_dim, out_dim, dropout=0.1, residual=True, **kwargs):
        super().__init__(**kwargs)
        self.A = nn.Linear(in_dim, out_dim, bias=True)
        self.B = nn.Linear(in_dim, out_dim, bias=True)
        self.C = nn.Linear(in_dim, out_dim, bias=True)
        self.D = nn.Linear(in_dim, out_dim, bias=True)
        self.E = nn.Linear(in_dim, out_dim, bias=True)

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, e, edge_index):
        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        x, e = self.propagate(edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, e=e, Ax=Ax)

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = F.relu(x)
        e = F.relu(e)

        if self.residual:
            x = x_in + x
            e = e_in + e

        x = F.dropout(x, self.dropout, training=self.training)
        edge_attr = F.dropout(e, self.dropout, training=self.training)

        return x, edge_attr

    def message(self, Dx_i, Ex_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """

        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)
        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """

        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size, reduce="sum")

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size, reduce="sum")

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """

        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


class GatedGCNLSPELayer(pyg_nn.MessagePassing):
    """
    GatedGCN layer
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """

    def __init__(self, in_dim, out_dim, dropout=0.1, residual=True, **kwargs):
        super().__init__(**kwargs)
        self.A = nn.Linear(in_dim * 2, out_dim, bias=True)
        self.B = nn.Linear(in_dim * 2, out_dim, bias=True)
        self.D = nn.Linear(in_dim * 2, out_dim, bias=True)
        self.E = nn.Linear(in_dim * 2, out_dim, bias=True)
        self.Ap = nn.Linear(in_dim, out_dim, bias=True)
        self.Bp = nn.Linear(in_dim, out_dim, bias=True)
        self.Dp = nn.Linear(in_dim, out_dim, bias=True)
        self.Ep = nn.Linear(in_dim, out_dim, bias=True)
        self.C = nn.Linear(in_dim, out_dim, bias=True)

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual

        self.e = None
        self.sigma_ij = None

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, e, edge_index, pe):
        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            e_in = e
            pe_in = pe

        x_pe = torch.cat((x, pe), dim=-1)
        Ax = self.A(x_pe)
        Bx = self.B(x_pe)
        Dx = self.D(x_pe)
        Ex = self.E(x_pe)
        Ce = self.C(e)
        Ape = self.Ap(pe)
        Bpe = self.Bp(pe)
        Dpe = self.Dp(pe)
        Epe = self.Ep(pe)

        x, e = self.propagate(edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, e=e, Ax=Ax, is_pe=False)
        pe, _ = self.propagate(edge_index, Bx=Bpe, Dx=Dpe, Ex=Epe, Ce=Ce, e=e, Ax=Ape, is_pe=True)

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = F.relu(x)
        e = F.relu(e)
        pe = torch.tanh(pe)

        if self.residual:
            x, e, pe = x_in + x, e_in + e, pe_in + pe

        x = F.dropout(x, self.dropout, training=self.training)
        edge_attr = F.dropout(e, self.dropout, training=self.training)
        pe = F.dropout(pe, self.dropout, training=self.training)

        return x, pe

    def message(self, Dx_i, Ex_j, Ce, is_pe):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """

        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)
        self.e = e_ij

        if not is_pe:
            self.sigma_ij = sigma_ij

        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx, is_pe):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        if is_pe:
            sigma_ij = self.sigma_ij

        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size, reduce="sum")

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size, reduce="sum")

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)

        if is_pe:
            del self.sigma_ij

        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """

        x = Ax + aggr_out
        e_out = self.e

        del self.e

        return x, e_out


## Implementation using R-GCN of newest pyg ======================================


class RGatedGCNLayer(pyg_nn.MessagePassing):
    """Implementation for Relational Gated Graph ConvNets.
    Currently, only Basis-decomposition regularization is usable

    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        dropout: float = 0.1,
        residual: bool = True,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks

        self.weight = nn.Parameter(torch.Tensor(num_bases, out_channels, out_channels))
        self.comp = nn.Parameter(torch.Tensor(num_relations, num_bases))

        self.A = nn.Linear(in_channels, out_channels, bias=True)
        self.B = nn.Linear(in_channels, out_channels, bias=True)
        self.D = nn.Linear(in_channels, out_channels, bias=True)
        self.E = nn.Linear(in_channels, out_channels, bias=True)

        self.bn_node_x = nn.BatchNorm1d(out_channels)
        self.dropout = dropout
        self.residual = residual

        self.register_parameter("root", None)
        self.register_parameter("bias", None)

        ## Initialize
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)

    def message(self, Dx_i, Ex_j):
        """
        {}x_i           : [n_edges, out_channels]
        {}x_j           : [n_edges, out_channels]
        {}e             : [n_edges, out_channels]
        """

        e_ij = Dx_i + Ex_j
        sigma_ij = torch.sigmoid(e_ij)

        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_channels]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_channels]
        """

        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        # # fmt: off
        # import ipdb; ipdb.set_trace()
        # # fmt: on
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size, reduce="sum")

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size, reduce="sum")

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)

        return out

    def forward(self, x, edge_index, edge_type):
        r"""
        Args:
            x: The input node features.
            edge_type: The one-dimensional relation type/index for each edge in

        """

        device = x.device

        if self.residual:
            x_in = x

        ##########################################
        ## Firstly project to same output dim
        ##########################################
        Ax, Bx, Dx, Ex = self.A(x), self.B(x), self.D(x), self.E(x)

        ##########################################
        ## Start propagation
        ##########################################
        out = torch.zeros(x.size(0), self.out_channels, device=device)
        weight = self.weight
        if self.num_bases is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.out_channels, self.out_channels
            )

        for i in range(self.num_relations):
            masked_edge_indx = masked_edge_index(edge_index, edge_type == i)

            h = self.propagate(masked_edge_indx, Bx=Bx, Dx=Dx, Ex=Ex, Ax=Ax)
            out = out + (h @ weight[i])

        ##########################################
        ## Apply BN, activation and residual
        ##########################################
        x = self.bn_node_x(out)
        x = torch.relu(x)

        # if self.residual:
        #     x = x_in + x

        x = F.dropout(x, self.dropout, training=self.training)

        ##########################################
        ## output
        ##########################################

        return x

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_channels] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_channels]
        """

        x = Ax + aggr_out

        return x


class FastRGatedGCNLayer(RGatedGCNLayer):
    def forward(self, x, edge_index, edge_type):
        r"""
        Args:
            x: The input node features.
            edge_type: The one-dimensional relation type/index for each edge in

        """

        if self.residual:
            x_in = x

        ##########################################
        ## Firstly project to same output dim
        ##########################################
        Ax, Bx, Dx, Ex = self.A(x), self.B(x), self.D(x), self.E(x)

        ##########################################
        ## Start propagation
        ##########################################
        out = self.propagate(edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ax=Ax, edge_type=edge_type)

        ##########################################
        ## Apply BN, activation and residual
        ##########################################
        x = self.bn_node_x(out)
        x = torch.relu(x)

        if self.residual:
            x = x_in + x

        x = F.dropout(x, self.dropout, training=self.training)

        ##########################################
        ## output
        ##########################################

        return x

    def message(self, Dx_i, Ex_j, edge_type):
        e_ij = Dx_i + Ex_j
        sigma_ij = torch.sigmoid(e_ij)

        weight = self.weight
        if self.num_bases is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.out_channels, self.out_channels
            )

        return torch.bmm(sigma_ij.unsqueeze(-2), weight[edge_type]).squeeze(-2)

    def aggregate(self, sigma_ij, index, Bx_j, Bx, dim_size, edge_type):
        out = super().aggregate(sigma_ij, index, Bx_j, Bx)

        # # fmt: off
        # import ipdb; ipdb.set_trace()
        # # fmt: on

        # return scatter(out, index, dim=self.node_dim, dim_size=dim_size)

        return out


class NewRGCNConv(pyg_nn.MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    .. note::
        This implementation is as memory-efficient as possible by iterating
        over each individual relation type.
        Therefore, it may result in low GPU utilization in case the graph has a
        large number of relations.
        As an alternative approach, :class:`FastRGCNConv` does not iterate over
        each individual type, but may consume a large amount of memory to
        compensate.
        We advise to check out both implementations to see which one fits your
        needs.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
            In case no input features are given, this argument should
            correspond to the number of nodes in your graph.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int, optional): If set, this layer will use the
            basis-decomposition regularization scheme where :obj:`num_bases`
            denotes the number of bases to use. (default: :obj:`None`)
        num_blocks (int, optional): If set, this layer will use the
            block-diagonal-decomposition regularization scheme where
            :obj:`num_blocks` denotes the number of blocks to use.
            (default: :obj:`None`)
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        aggr: str = "mean",
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", aggr)
        super().__init__(node_dim=0, **kwargs)

        if num_bases is not None and num_blocks is not None:
            raise ValueError(
                "Can not apply both basis-decomposition and "
                "block-diagonal-decomposition at the same time."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        if num_bases is not None:
            self.weight = Parameter(torch.Tensor(num_bases, in_channels[0], out_channels))
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))
            self.att = self.comp
            self.basis = self.weight

        elif num_blocks is not None:
            assert in_channels[0] % num_blocks == 0 and out_channels % num_blocks == 0
            self.weight = Parameter(
                torch.Tensor(
                    num_relations,
                    num_blocks,
                    in_channels[0] // num_blocks,
                    out_channels // num_blocks,
                )
            )
            self.register_parameter("comp", None)

        else:
            self.weight = Parameter(torch.Tensor(num_relations, in_channels[0], out_channels))
            self.register_parameter("comp", None)

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter("root", None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)

    def forward(
        self,
        x: Union[OptTensor, Tuple[OptTensor, Tensor]],
        edge_index: Adj,
        edge_type: OptTensor = None,
    ):
        r"""
        Args:
            x: The input node features. Can be either a :obj:`[num_nodes,
                in_channels]` node feature matrix, or an optional
                one-dimensional node index tensor (in which case input features
                are treated as trainable node embeddings).
                Furthermore, :obj:`x` can be of type :obj:`tuple` denoting
                source and destination node features.
            edge_index (LongTensor or SparseTensor): The edge indices.
            edge_type: The one-dimensional relation type/index for each edge in
                :obj:`edge_index`.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                :class:`torch_sparse.tensor.SparseTensor`.
                (default: :obj:`None`)
        """

        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels
            )

        if self.num_blocks is not None:  # Block-diagonal-decomposition =====

            if x_l.dtype == torch.long and self.num_blocks is not None:
                raise ValueError(
                    "Block-diagonal decomposition not supported "
                    "for non-continuous input features."
                )

            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x_l, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum("abc,bcd->abd", h, weight[i])
                out += h.contiguous().view(-1, self.out_channels)

        else:  # No regularization/Basis-decomposition ========================
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)

                if x_l.dtype == torch.long:
                    out += self.propagate(tmp, x=weight[i, x_l], size=size)
                else:
                    h = self.propagate(tmp, x=x_l, size=size)
                    out = out + (h @ weight[i])

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, num_relations={self.num_relations})"
        )


class NewFastRGCNConv(NewRGCNConv):
    r"""See :class:`RGCNConv`."""

    def forward(
        self,
        x: Union[OptTensor, Tuple[OptTensor, Tensor]],
        edge_index: Adj,
        edge_type: OptTensor = None,
    ):
        """"""
        self.fuse = False
        assert self.aggr in ["add", "sum", "mean"]

        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        # propagate_type: (x: Tensor, edge_type: OptTensor)
        out = self.propagate(edge_index, x=x_l, edge_type=edge_type, size=size)

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_type: Tensor, edge_index_j: Tensor) -> Tensor:
        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels
            )

        if self.num_blocks is not None:  # Block-diagonal-decomposition =======
            if x_j.dtype == torch.long:
                raise ValueError(
                    "Block-diagonal decomposition not supported "
                    "for non-continuous input features."
                )

            weight = weight[edge_type].view(-1, weight.size(2), weight.size(3))
            x_j = x_j.view(-1, 1, weight.size(1))
            return torch.bmm(x_j, weight).view(-1, self.out_channels)

        else:  # No regularization/Basis-decomposition ========================
            if x_j.dtype == torch.long:
                weight_index = edge_type * weight.size(1) + edge_index_j
                return weight.view(-1, self.out_channels)[weight_index]

            return torch.bmm(x_j.unsqueeze(-2), weight[edge_type]).squeeze(-2)

    def aggregate(
        self, inputs: Tensor, edge_type: Tensor, index: Tensor, dim_size: Optional[int] = None
    ) -> Tensor:

        # Compute normalization in separation for each `edge_type`.
        if self.aggr == "mean":
            norm = F.one_hot(edge_type, self.num_relations).to(torch.float)
            norm = scatter(norm, index, dim=0, dim_size=dim_size)[index]
            norm = torch.gather(norm, 1, edge_type.view(-1, 1))
            norm = 1.0 / norm.clamp_(1.0)
            inputs = norm * inputs

        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size)


class NewRGCNConvLSPE(NewRGCNConv):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        aggr: str = "mean",
        root_weight: bool = True,
        bias: bool = True,
        is_residual: bool = False,
        **kwargs,
    ):

        self.residual = is_residual

        if self.residual:
            in_channels = out_channels

        super().__init__(
            in_channels,
            out_channels,
            num_relations,
            num_bases,
            num_blocks,
            aggr,
            root_weight,
            bias,
            **kwargs,
        )

        if self.residual:
            self.lin_x = nn.Linear(in_channels, out_channels)
            self.lin_pe = nn.Linear(in_channels, out_channels)
        self.lin_x_pe_cat = nn.Linear(in_channels * 2, out_channels)
        self.bn_node_x = nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def forward(
        self,
        x: Tensor,
        pe: Tensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
    ):

        if self.residual:
            x, pe = self.lin_x(x), self.lin_pe(pe)
            x_in, pe_in = x, pe

        x = self.lin_x_pe_cat(torch.cat((x, pe), dim=-1))
        updated_x = super().forward(x, edge_index, edge_type)
        updated_x = F.relu(self.bn_node_x(x))
        updated_pe = super().forward(pe, edge_index, edge_type)
        updated_pe = torch.tanh(updated_pe)

        x, pe = updated_x, updated_pe

        if self.residual:
            x, pe = x_in + x, pe_in + pe

        return x, pe


class NewFastRGCNConvLSPE(NewFastRGCNConv):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        aggr: str = "mean",
        root_weight: bool = True,
        bias: bool = True,
        is_residual: bool = False,
        **kwargs,
    ):

        self.residual = is_residual

        if self.residual:
            in_channels = out_channels

        super().__init__(
            in_channels,
            out_channels,
            num_relations,
            num_bases,
            num_blocks,
            aggr,
            root_weight,
            bias,
            **kwargs,
        )

        if self.residual:
            self.lin_x = nn.Linear(in_channels, out_channels)
            self.lin_pe = nn.Linear(in_channels, out_channels)
        self.lin_x_pe_cat = nn.Linear(in_channels * 2, out_channels)
        self.bn_node_x = nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def forward(
        self,
        x: Tensor,
        pe: Tensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
    ):

        if self.residual:
            x, pe = self.lin_x(x), self.lin_pe(pe)
            x_in, pe_in = x, pe

        x = self.lin_x_pe_cat(torch.cat((x, pe), dim=-1))
        updated_x = super().forward(x, edge_index, edge_type)
        updated_x = F.relu(self.bn_node_x(x))
        updated_pe = super().forward(pe, edge_index, edge_type)
        updated_pe = torch.tanh(updated_pe)

        x, pe = updated_x, updated_pe

        if self.residual:
            x, pe = x_in + x, pe_in + pe

        return x, pe


## Implementation using R-GCN of pyg 1.4.2 ======================================


class OldRGCNConvLSPE(pyg_nn.conv.RGCNConv):
    """R-GCN-LSPE based on implement of R-GCN of pyg ver 1.4.2"""

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        root_weight: bool = True,
        bias: bool = True,
        is_residual: bool = False,
        **kwargs,
    ):

        self.residual = is_residual

        super().__init__(
            out_channels,
            out_channels,
            num_relations,
            num_bases,
            root_weight,
            bias,
            **kwargs,
        )

        self.lin_x = nn.Linear(in_channels, out_channels)
        self.lin_pe = nn.Linear(in_channels, out_channels)
        self.lin_x_pe_cat = nn.Linear(out_channels * 2, out_channels)
        self.bn_node_x = nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def forward(
        self,
        x: Tensor,
        pe: Tensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
    ):

        x, pe = self.lin_x(x), self.lin_pe(pe)

        if self.residual:
            x_in, pe_in = x, pe

        x = self.lin_x_pe_cat(torch.cat((x, pe), dim=-1))
        updated_x = super().forward(x, edge_index, edge_type)
        updated_x = F.relu(self.bn_node_x(x))
        updated_pe = super().forward(pe, edge_index, edge_type)
        updated_pe = torch.tanh(updated_pe)

        x, pe = updated_x, updated_pe

        if self.residual:
            x, pe = x_in + x, pe_in + pe

        return x, pe


## ==============================================================================


class GNNModule(torch.nn.Module):
    # a base GNN class, GCN message passing + sum_pooling
    def __init__(
        self,
        gconv=GatedGCNLSPELayer,
        input_dim=64,
        latent_dim=64,
        num_layers=4,
        dropout=0.1,
        residual=True,
    ):
        super(GNNModule, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            gconv(in_dim=input_dim, out_dim=latent_dim, dropout=dropout, residual=residual)
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                gconv(in_dim=latent_dim, out_dim=latent_dim, dropout=dropout, residual=residual)
            )

        ## Init module's params
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.initialize_weights()

    def forward(self, x, e, edge_index, pe):
        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        pe              : [n_nodes, in_dim]
        """

        for conv in self.convs:
            x, e, edge_index, pe = conv(x, e, edge_index, pe)

        return x, e

    def __repr__(self):
        return self.__class__.__name__
