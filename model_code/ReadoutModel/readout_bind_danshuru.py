import torch
import torch.nn as nn
import dgl
from dgllife.model.readout import AttentiveFPReadout
from ReadoutModel.readout_my import MyReadout
from DMPNN_res.Layers import DMPNN_Encoder
from EAT.Layer_edge_v import EAT_
from SIGN.Layers_withoutattention_andangle import SIGN
from Final.final import Bind
from utilis.function import get_activation_func
from dgllife.model.gnn.gcn import GCN
from torch_sparse import SparseTensor


class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation='ReLU', bias=True):
        super(DenseLayer, self).__init__()
        if activation is not None:
            self.act = get_activation_func(activation)
        else:
            self.act = None
        if not bias:
            self.fc = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, input_feat):
        if self.act is not None:
            return self.act(self.fc(input_feat))
        else:
            return self.fc(input_feat)


def distance(edge_feat):
    def func(edges):
        return {'dist': (edges.src[edge_feat] - edges.dst[edge_feat]).norm(dim=-1)}

    return func


# def distance_emb(edge_feat, device):
#     def func(edges):
#         return {'rbf': _rbf(edges.data[edge_feat],device=device)}
#
#     return func

def edge_cat(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: torch.cat([edges.data[src_field], edges.data[dst_field]], dim=-1)}

    return func


class DMPNN(nn.Module):
    def __init__(self, hidden_dim, radius, T, p_dropout, ffn_num_layers, num_heads=4,
                 in_dim_atom=70, in_dim_edge=14, encoder_type="DMPNN_res", readout_type="AttFP", output_dim=1,
                 degree_information=0, GCN_=0,
                 cs=0):
        super(DMPNN, self).__init__()

        self.encoder_type = encoder_type
        self.readout_type = readout_type
        self.degree_information = degree_information
        self.GCN_ = GCN_
        self.cs = cs
        # self.mu = nn.Parameter(torch.tensor([2.0], dtype=torch.float32), requires_grad=False)
        # self.v = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=False)

        if encoder_type == "DMPNN_res":
            self.encoder = DMPNN_Encoder(hidden_dim=hidden_dim, radius=radius, p_dropout=p_dropout)
        elif encoder_type == "Bind":
            self.encoder = Bind(num_head=num_heads, feat_drop=p_dropout, attn_drop=p_dropout,
                                num_convs=radius, hidden_dim=hidden_dim, activation="ReLU")
        elif encoder_type == "SIGN":
            self.encoder = SIGN()
        elif encoder_type == "EAT":
            self.encoder = EAT_(hidden_dim, num_heads, n_layers=4, dropout=0.2, layer_norm=True,
                                batch_norm=False, residual=True, use_bias=True)

        if readout_type == "AttFP":

            self.readout = AttentiveFPReadout(feat_size=hidden_dim,
                                              num_timesteps=T,
                                              dropout=p_dropout)
        elif readout_type == "My":
            self.readout = MyReadout(hidden_dim=hidden_dim,
                                     num_timesteps=T,
                                     dropout=p_dropout)

        self.hidden_dim = hidden_dim
        self.bias = False

        self.act_func = get_activation_func("ReLU")
        self.dropout = p_dropout
        self.num_FFN_layer = ffn_num_layers

        self.output_dim = output_dim

        ffn = [nn.Linear(hidden_dim, int(hidden_dim * 0.5)), nn.ReLU(),
               nn.Linear(int(hidden_dim * 0.5), int(hidden_dim * 0.25)), nn.ReLU(),
               nn.Linear(int(hidden_dim * 0.25), int(hidden_dim * 0.125)), nn.ReLU(),
               nn.Linear(int(hidden_dim * 0.125), self.output_dim)]

        self.FNN = nn.Sequential(*ffn)

        self.GCN = GCN(in_feats=70, hidden_feats=[hidden_dim, hidden_dim, hidden_dim],
                       batchnorm=[True, True, True])

        self.lin_atom1 = DenseLayer(in_dim=in_dim_atom, out_dim=hidden_dim, bias=True)
        self.lin_edge1 = DenseLayer(in_dim=in_dim_edge, out_dim=int(hidden_dim / 2), bias=True)
        # self.lin_rbf = DenseLayer(in_dim=16, out_dim=int(hidden_dim / 2))

        # self.ln = nn.LayerNorm(hidden_dim)
        self.degree_encoder = nn.Embedding(200, hidden_dim, padding_idx=0)

        # cross attention
        self.down_lin1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

    def triplets(self, g):
        row, col = g.edges()  # j --> i
        num_nodes = g.num_nodes()

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return idx_kj, idx_ji

    # def mu_and_v(self):
    #     self.mu.requires_grad_()
    #     self.v.requires_grad_()

    def forward(self, g1,g_pocket=None):

        index_kj1, index_ji1 = self.triplets(g1)
        k1 = torch.nonzero(g1.ndata["p_or_l"], as_tuple=True)[0]  # 口袋原子索引
        k1_ = torch.nonzero(g1.ndata["p_or_l"] == 0, as_tuple=True)[0]  # 配体原子索引

        if self.GCN_ == 1:  # 打开
            g_pocket = dgl.add_self_loop(g_pocket)
            g_pocket.ndata["atom_feature_h"] = self.GCN(g_pocket, g_pocket.ndata["atom_feature"].to(torch.float32))
            g1.ndata["atom_feature_h"] = torch.zeros([g1.ndata["atom_feature"].shape[0], self.hidden_dim]).to(
                device=g1.device)
            g1.ndata["atom_feature_h"][k1_] = self.lin_atom1(
                g1.ndata["atom_feature"].to(torch.float32)[k1_])  # 70 --> 200
            g1.ndata["atom_feature_h"][k1] = g_pocket.ndata["atom_feature_h"]  # 卷积之后的口袋信息更新现有信息

        if self.GCN_ == 0:  # 关闭
            g1.ndata["atom_feature_h"] = self.lin_atom1(g1.ndata["atom_feature"].to(torch.float32))  # 70 --> 200


        if self.degree_information == 1:
            g1.ndata["atom_feature_h"] = g1.ndata["atom_feature_h"] + self.degree_encoder(g1.in_degrees())  # 度信息

        g1.edata["edge_feature_h"] = self.lin_edge1(g1.edata["edge_feature"].to(torch.float32))  # 14 --> 100


        g1.apply_edges(distance('atom_coordinate'))  # 获得dist


        # 取log并*2
        diss1 = torch.where(g1.edata['attention_weight'] == 0,
                            torch.tensor(-1).to(device=g1.device, dtype=torch.float32),
                            torch.log(g1.edata['attention_weight']) * 2)

        # 共价边恢复成1
        g1.edata['dist_decay'] = torch.where(g1.edata['attention_weight'] == 1,
                                             torch.tensor(1.0).to(device=g1.device, dtype=torch.float32),
                                             diss1)


        h1, att1 = self.encoder(g1, index_kj1, index_ji1)

        with g1.local_scope():

            g1.ndata['h'] = h1
            d1 = g1.in_degrees()[k1_].unsqueeze(dim=-1)
            g1.remove_nodes(k1)

            if self.readout_type == "AttFP":
                hsg1, a1 = self.readout(g1, g1.ndata['h'], True)

            elif self.readout_type == "My":
                hsg1 = self.readout(g1, g1.ndata['h'], d1)

            elif self.readout_type == "mean":
                hsg1 = dgl.readout_nodes(g1, 'h', op='mean')
            elif self.readout_type == "sum":
                hsg1 = dgl.readout_nodes(g1, 'h', op='sum')


            # hsg1 = torch.cat([hsg1, dgl.readout_nodes(g1, 'interaction_embedding', op='mean')], dim=-1)  # 非键相互作用力信息
            # hsg2 = torch.cat([hsg2, dgl.readout_nodes(g2, 'interaction_embedding', op='mean')], dim=-1)

            zk = self.FNN(hsg1)

        return zk, att1, a1