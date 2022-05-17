from all_packages import *
from hypermixer import HyperMixerLayer
from regularization.mlp import MLP
from regularization.models import ContrastiveModel
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import dropout_adj
from util_functions import *

from layers import *


class IGMC(GNN):
    # The GNN model of Inductive Graph-based Matrix Completion.
    # Use RGCN convolution + center-nodes readout.
    def __init__(
        self,
        dataset,
        gconv=None,
        latent_dim=None,
        num_relations=5,
        num_bases=2,
        regression=False,
        adj_dropout=0.2,
        force_undirected=False,
        side_features=False,
        n_side_features=0,
        multiply_by=1,
        n_nodes=3000,
        class_values=None,
        dropout=0.2,
        args=None,
    ):
        # gconv = GatedGCNLayer GatedGCNLSPELayer RGatedGCNLayer RGCNConv
        if args.scenario in [1, 2, 3, 4, 5, 6, 7, 8]:
            # gconv = RGCNConvLSPE GatedGCNLSPELayer NewFastRGCNConv OldRGCNConvLSPE
            gconv = OldRGCNConvLSPE
        elif args.scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
            gconv = GatedGCNLSPELayer
        else:
            raise NotImplementedError()

        super(IGMC, self).__init__(
            dataset,
            gconv,
            latent_dim,
            regression,
            adj_dropout,
            force_undirected,
            num_relations,
            num_bases,
        )

        if latent_dim is None:
            latent_dim = [32, 32, 32, 32]

        self.multiply_by = multiply_by
        self.class_values = class_values
        self.gconv = gconv
        self.side_features = side_features
        self.scenario = args.scenario
        self.mode = args.mode
        self.metric = args.metric
        self.dataname = args.data_name
        self.pe_dim = args.pe_dim
        self.arr = args.ARR

        ## Declare modules to convert node feat, pe to hidden vectors
        self.node_feat_dim = dataset.num_features - self.pe_dim
        if self.scenario in [1, 2, 3, 4, 5, 6, 7, 8]:
            self.lin_pe = nn.Linear(self.pe_dim, self.node_feat_dim)
        elif self.scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
            self.lin_pe = nn.Linear(self.pe_dim, latent_dim[0])
            self.edge_embd = nn.Linear(1, latent_dim[0])
            self.lin_x = nn.Linear(self.node_feat_dim, latent_dim[0])

        ## Declare GNN layers
        self.convs = torch.nn.ModuleList()
        if self.scenario in [1, 2, 3, 4, 5, 6, 7, 8]:
            # kwargs = {"num_relations": num_relations, "num_bases": num_bases, "is_residual": True}
            kwargs = {"num_relations": num_relations, "num_bases": num_bases}
            self.convs.append(gconv(self.node_feat_dim, latent_dim[0], **kwargs))
            for i in range(0, len(latent_dim) - 1):
                self.convs.append(gconv(latent_dim[i], latent_dim[i + 1], **kwargs))
        elif self.scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
            self.convs.append(gconv(latent_dim[0], latent_dim[0]))
            for i in range(0, len(latent_dim) - 1):
                self.convs.append(gconv(latent_dim[i], latent_dim[i + 1]))

        ## Declare Mixer: TransEncoder ; HyperMixer
        if args.mixer == "trans_encoder":
            self.mixer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=4, nhead=2), 4)
        elif self.mixer == "hyper_mixer":
            self.mixer = nn.Sequential(*[HyperMixerLayer(N=420, hid_dim=4) for _ in range(4)])
        else:
            raise NotImplementedError()

        ## Define final FF
        if self.scenario in [1, 2, 3, 4, 9, 10, 11, 12]:
            final_dim = 2 * sum(latent_dim)
        elif self.scenario in [5, 6, 7, 8, 13, 14, 15, 16]:
            final_dim = 4 * sum(latent_dim)
        else:
            raise NotImplementedError()
        ## NOTE: If using PE as node feat, enable the following
        # final_dim = 2 * sum(latent_dim)
        self.final_ff = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(final_dim, 1),
        )

        self.graphsizenorm = GraphSizeNorm()

        ## init weights
        self.initialize_weights()

        ## Load weights of Contrastive MLP layer
        if self.mode == "coop":
            self.load_pretrained_weights()
            self.contrastive_model = ContrastiveModel(d_pe=self.pe_dim, metric=self.metric)
            self.mlp_contrastive = self.contrastive_model.mlp

    def save_pretrained_weights(self):
        """Save weights of gconv layers and mixer"""
        path_pretrained = f"weights/{self.dataname}/{self.scenario}_{self.pe_dim}_{self.gconv.__name__}_{type(self.mixer).__name__}.pkl"

        pretrained_weights = {"convs": None, "mixer": None}
        pretrained_weights["mixer"] = self.mixer.state_dict()
        pretrained_weights["convs"] = [conv.state_dict() for conv in self.convs]

        os.makedirs(osp.dirname(path_pretrained), exist_ok=True)
        torch.save(pretrained_weights, path_pretrained)

    def load_pretrained_weights(self):
        """Load weights of gconv layers and mixer"""

        path_pretrained = f"weights/{self.dataname}/{self.scenario}_{self.pe_dim}_{self.gconv.__name__}_{type(self.mixer).__name__}.pkl"
        if not osp.isfile(path_pretrained):
            logger.error(f"Path to pretrained GNN and Mixer not valid: {path_pretrained}")
            exit(1)

        ## Load gconvs and mixer
        logger.info(f"Load pretrained weights from: {path_pretrained}")

        pretrained_weights = torch.load(path_pretrained)
        self.mixer.load_state_dict(pretrained_weights["mixer"])
        for ith, conv in enumerate(self.convs):
            conv.load_state_dict(pretrained_weights["convs"][ith])

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (GatedGCNLayer, GatedGCNLSPELayer)):
                m.initialize_weights()

    def conv_score2class(self, score, device=None):
        if not isinstance(self.class_values, torch.Tensor):
            self.class_values = torch.tensor(self.class_values, device=device)

        classes_ = self.class_values.unsqueeze(0).repeat(score.shape[0], 1)
        indices = torch.abs((score - classes_)).argmin(-1)

        return indices

    def create_trans_mask(self, batch, dtype, device, batch_size=50):
        masks = []
        for i in range(batch_size):
            n_nodes_batch = torch.sum(batch == i)
            mask_batch = torch.ones((n_nodes_batch, n_nodes_batch), dtype=dtype, device=device)
            masks.append(mask_batch)

        mask = torch.block_diag(*masks)

        return mask

    def conv2tensor(self, permuted_graphs: list, device=None, dtype=None):
        x, trg = [], []
        for p in permuted_graphs:
            x += [torch.from_numpy(_[0]).to(device=device, dtype=dtype) for _ in p[0]]
            trg += [torch.from_numpy(_[1]).to(device=device, dtype=dtype) for _ in p[0]]
        # x: list of [n*, d]
        # trg: list of [n*, n*]

        n_max = max([_.shape[0] for _ in x])
        d = x[0].shape[-1]

        X, trgs, mask = [], None, []

        ## Create mask
        mask = torch.tensor([x_.shape[0] for x_ in x])
        # [bz]

        ## Create X
        for x_ in x:
            pad0 = torch.zeros((n_max - x_.shape[0], d), device=device, dtype=dtype)
            x_ = torch.cat((x_, pad0), dim=0)
            X.append(x_)
        X = torch.stack(X)
        # [bz, n_max, d]

        trgs = torch.block_diag(*trg)
        # [N, N]

        return X, trgs, mask

    def forward(self, data):

        x, edge_index, edge_type, batch, permuted_graphs = (
            data.x,
            data.edge_index,
            data.edge_type,
            data.batch,
            data.permuted_graphs,
        )

        device, dtype = x.device, x.dtype

        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index,
                edge_type,
                p=self.adj_dropout,
                force_undirected=self.force_undirected,
                num_nodes=len(x),
                training=self.training,
            )

        #########################################################
        ## 1. Calculate Contrastive Learning loss
        #########################################################
        loss_mse_contrs = 0
        if self.mode == "coop":
            X, trgs, mask = self.conv2tensor(permuted_graphs, device=device, dtype=dtype)

            loss_mse_contrs = self.contrastive_model(X=X, trgs=trgs, mask=mask)

        #########################################################
        ## 2. Pass thru GNN
        #########################################################
        ## Extract node feature, RWPE info and global node index
        node_subgraph_feat, pe = (
            x[:, : self.node_feat_dim],
            x[:, self.node_feat_dim :],
        )

        if isinstance(self.mixer, nn.TransformerEncoder):
            mask = self.create_trans_mask(batch, x.dtype, x.device)
            node_subgraph_feat = self.mixer(node_subgraph_feat.unsqueeze(1), mask).squeeze(1)
        else:
            for hyper_mixer in self.mixer:
                node_subgraph_feat = hyper_mixer(node_subgraph_feat, batch)

        ## Convert node feat, pe to suitable dim before passing thu GNN layers
        if self.mode == "coop":
            pe = self.lin_pe(self.mlp_contrastive(pe))
        else:
            pe = self.lin_pe(pe)
        x = node_subgraph_feat
        # NOTE: If using PE as node_feat, enable the following
        # x = pe

        if self.scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
            edge_type = edge_type.unsqueeze(-1).float()
            edge_embd = self.edge_embd(edge_type)
            x = self.lin_x(x)

        ## Apply graph size norm
        if self.scenario in [3, 4, 7, 8, 11, 12, 15, 16]:
            x = self.graphsizenorm(x, batch)

        ## Pass node feat thru GNN layers
        concat_states = []
        for conv in self.convs:
            if self.scenario in [1, 2, 3, 4, 5, 6, 7, 8]:
                # NOTE: If using PE as node_feat, enable the following
                # x = conv(x, edge_index, edge_type)
                x, pe = conv(x, pe, edge_index, edge_type)
            elif self.scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
                # NOTE: If using PE as node_feat, enable the following
                # x, edge_embd = conv(x, edge_embd, edge_index)
                x, edge_embd, pe = conv(x, edge_embd, edge_index, pe)
            else:
                raise NotImplementedError()

            if self.scenario in [5, 6, 7, 8, 13, 14, 15, 16]:
                # NOTE: If using PE as node_feat, enable the following
                # concat_states.append(x)
                concat_states.append(torch.cat((x, pe), dim=-1))
            elif self.scenario in [1, 2, 3, 4, 9, 10, 11, 12]:
                concat_states.append(x)
            else:
                raise NotImplementedError()

        concat_states = torch.cat(concat_states, 1)

        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1
        x = torch.cat([concat_states[users], concat_states[items]], 1)
        if self.side_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        x = self.final_ff(x)[:, 0]

        #########################################################
        ## 3. Calculate loss
        #########################################################
        loss = 0

        ## Loss MSE from GNN: relation prediction
        loss_mse_GNN = F.mse_loss(x, data.y.view(-1))

        ## Loss reguarization ARR
        if self.scenario in [1, 2, 3, 4, 5, 6, 7, 8]:
            for gconv in self.convs:
                w = torch.matmul(gconv.att, gconv.basis.view(gconv.num_bases, -1)).view(
                    gconv.num_relations, gconv.in_channels, gconv.out_channels
                )
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :]) ** 2)
        elif self.scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
            w = self.edge_embd.weight
            reg_loss = torch.sum((w[1:] - w[:-1]) ** 2)
        loss_reg = self.arr * reg_loss

        loss = loss_mse_contrs + loss_mse_GNN + loss_reg

        return loss, x
