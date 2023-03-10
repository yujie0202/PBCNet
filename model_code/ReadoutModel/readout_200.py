import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import dgl
from dgllife.model.readout import AttentiveFPReadout

from DMPNN_res.Layers import DMPNN_Encoder
from EAT.Layer import EAT_
from SIGN.Layers_withoutattention_andangle import SIGN
from utilis.function import get_activation_func
from dgllife.model.gnn.gcn import GCN


def atom_cross_att(g1, g2):
    t = Variable(torch.tensor(math.sqrt(1 / 200)), requires_grad=True)
    batch_size = g1.batch_size

    for_g1 = []
    for_g2 = []
    for i in range(batch_size):
        hg1 = dgl.slice_batch(g1, i).ndata['h']  # num_atoms * 200
        hg2 = dgl.slice_batch(g2, i).ndata['h']
        att = torch.matmul(hg1, torch.transpose(hg2, -1, -2)) * t  # num atoms A * num atoms B
        att_2 = nn.functional.softmax(att, dim=-1)
        att_1 = nn.functional.softmax(att, dim=-2)
        for_hg1 = torch.matmul(att_2, hg2)  # num atoms A * 200
        for_hg2 = torch.matmul(torch.transpose(att_1, -1, -2), hg1)  # num atoms B * 200
        for_g1.append(for_hg1)
        for_g2.append(for_hg2)

    h1 = torch.cat(for_g1, dim=0)
    h2 = torch.cat(for_g2, dim=0)

    return h1, h2


def _rbf(D, D_min=0., D_max=5., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation='ReLU', bias=True):
        super(DenseLayer, self).__init__()
        self.act = get_activation_func(activation)
        if not bias:
            self.fc = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, input_feat):
        return self.act(self.fc(input_feat))


def distance_emb(edge_feat, device):
    def func(edges):
        return {'rbf': _rbf((edges.src[edge_feat]-edges.dst[edge_feat]).norm(dim=-1),device=device)}

    return func


def edge_cat(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: torch.cat([edges.data[src_field], edges.data[dst_field]],dim=-1)}

    return func


class DMPNN(nn.Module):
    def __init__(self, hidden_dim, radius, T, p_dropout, ffn_num_layers, num_heads=4,
                 in_dim_atom=70, in_dim_edge=14,encoder_type="DMPNN_res", output_dim=1):
        super(DMPNN, self).__init__()

        self.encoder_type = encoder_type

        if encoder_type == "DMPNN_res":
            self.encoder = DMPNN_Encoder(hidden_dim=hidden_dim, radius=radius, p_dropout=p_dropout)
        elif encoder_type == "SIGN":
            self.encoder = SIGN()
        elif encoder_type == "EAT":
            self.encoder = EAT_(hidden_dim, num_heads, n_layers=3, dropout=0.2, layer_norm=True,
                                batch_norm=False, residual=True, use_bias=False)

        self.readout = AttentiveFPReadout(feat_size=hidden_dim,
                                          num_timesteps=T,
                                          dropout=p_dropout)
        self.hidden_dim = hidden_dim
        self.bias = False

        self.act_func = get_activation_func("ReLU")
        self.dropout = p_dropout
        self.num_FFN_layer = ffn_num_layers

        self.output_dim = output_dim

        ffn = [nn.Linear(hidden_dim * 3+18, 400), nn.ReLU()]
        ffn.append(nn.Linear(400, 200))
        ffn.append(nn.ReLU())
        ffn.append(nn.Linear(200, 100))
        ffn.append(nn.ReLU())
        ffn.append(nn.Linear(100, self.output_dim))

        self.FNN = nn.Sequential(*ffn)

        self.GCN = GCN(in_feats=70, hidden_feats=[200, 200, 200],
                       batchnorm=[False, False, False])

        self.lin_atom1 = DenseLayer(in_dim=in_dim_atom, out_dim=hidden_dim, bias=False)
        self.lin_edge1 = DenseLayer(in_dim=in_dim_edge, out_dim=100, bias=False)
        self.lin_rbf = DenseLayer(in_dim=16, out_dim=100)

        self.ln = nn.LayerNorm(200)
        self.degree_encoder = nn.Embedding(200, hidden_dim, padding_idx=0)

        # cross attention
        self.down_lin1 = nn.Linear(400, 200, bias=False)

    def forward(self, g_pocket, g1, g2):

        g_pocket = dgl.add_self_loop(g_pocket)

        g_pocket.ndata["atom_feature_h"] = self.GCN(g_pocket, g_pocket.ndata["atom_feature"].to(torch.float32))

        k1 = torch.nonzero(g1.ndata["p_or_l"], as_tuple=True)[0]  # ??????????????????
        k2 = torch.nonzero(g2.ndata["p_or_l"], as_tuple=True)[0]

        k1_ = torch.nonzero(g1.ndata["p_or_l"] == 0, as_tuple=True)[0]  # ??????????????????
        k2_ = torch.nonzero(g2.ndata["p_or_l"] == 0, as_tuple=True)[0]

        g1.ndata["atom_feature_h"] = torch.zeros([g1.ndata["atom_feature"].shape[0],200]).to(device=g1.device)
        g2.ndata["atom_feature_h"] = torch.zeros([g2.ndata["atom_feature"].shape[0],200]).to(device=g1.device)

        g1.ndata["atom_feature_h"][k1_] = self.lin_atom1(g1.ndata["atom_feature"].to(torch.float32)[k1_])  # 80 --> 200
        g2.ndata["atom_feature_h"][k2_] = self.lin_atom1(g2.ndata["atom_feature"].to(torch.float32)[k2_])

        g1.ndata["atom_feature_h"][k1] = g_pocket.ndata["atom_feature_h"]  # ?????????????????????????????????????????????
        g2.ndata["atom_feature_h"][k2] = g_pocket.ndata["atom_feature_h"]


        # g1.ndata["atom_feature_h"] = self.lin_atom1(g1.ndata["atom_feature"].to(torch.float32))  # 80 --> 200
        # g2.ndata["atom_feature_h"] = self.lin_atom1(g2.ndata["atom_feature"].to(torch.float32))
        g1.edata["edge_feature_h"] = self.lin_edge1(g1.edata["edge_feature"].to(torch.float32))  # 14 --> 100
        g2.edata["edge_feature_h"] = self.lin_edge1(g2.edata["edge_feature"].to(torch.float32))


        g1.apply_edges(distance_emb('atom_coordinate', g1.device))
        g2.apply_edges(distance_emb('atom_coordinate', g2.device))
        g1.edata["rbf"] = self.lin_rbf(g1.edata["rbf"].to(torch.float32))
        g2.edata["rbf"] = self.lin_rbf(g2.edata["rbf"].to(torch.float32))

        g1.apply_edges(edge_cat("edge_feature_h", "rbf", "edge_feature_h"))
        g2.apply_edges(edge_cat("edge_feature_h", "rbf", "edge_feature_h"))

        g1.ndata["atom_feature_h"] = self.ln(g1.ndata["atom_feature_h"])
        g2.ndata["atom_feature_h"] = self.ln(g2.ndata["atom_feature_h"])
        g1.edata["edge_feature_h"] = self.ln(g1.edata["edge_feature_h"])
        g2.edata["edge_feature_h"] = self.ln(g2.edata["edge_feature_h"])

        h1 = self.encoder(g1)
        h2 = self.encoder(g2)

        with g1.local_scope():
            with g2.local_scope():
                g1.ndata['h'] = h1
                g2.ndata['h'] = h2

                d1 = g1.in_degrees()[k1_]
                d2 = g2.in_degrees()[k2_]

                g1.remove_nodes(k1)
                g2.remove_nodes(k2)

                # atom cross docking
                h1_ca, h2_ca = atom_cross_att(g1, g2)

                g1.ndata['h'] = self.act_func(self.down_lin1(torch.cat([g1.ndata['h'], h1_ca], dim=-1)))
                g2.ndata['h'] = self.act_func(self.down_lin1(torch.cat([g2.ndata['h'], h2_ca], dim=-1)))

                g1.ndata['h'] = g1.ndata['h'] + self.degree_encoder(d1)  # ?????????
                g2.ndata['h'] = g2.ndata['h'] + self.degree_encoder(d2)
                hsg1 = self.readout(g1, g1.ndata['h'], False)
                hsg2 = self.readout(g2, g2.ndata['h'], False)

                hsg1 = torch.cat([hsg1, dgl.readout_nodes(g1, 'interaction_embedding', op='mean')],dim=-1)  # ???????????????????????????
                hsg2 = torch.cat([hsg2, dgl.readout_nodes(g2, 'interaction_embedding', op='mean')],dim=-1)

                zk = self.FNN(torch.cat([hsg1.to(torch.float32), hsg2.to(torch.float32), (hsg1 - hsg2).to(torch.float32)], dim=-1))
            
            return zk
