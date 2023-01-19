import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilis.function import get_activation_func


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


class GlobalPool(nn.Module):

    def __init__(self, hidden_dim, dropout):
        super(GlobalPool, self).__init__()
        self.num_head = 4
        self.act_att = get_activation_func("LeakyReLU")
        self.feat_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.out_feat = int(hidden_dim / self.num_head)
        self.hidden_dim = hidden_dim

        # self.k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, self.num_head, self.out_feat)))

        self.lin1 = DenseLayer(hidden_dim, hidden_dim, activation="ReLU")
        self.lin2 = DenseLayer(hidden_dim, hidden_dim, activation=None)

    def forward(self, g, node_feats, g_feats, degree):
        with g.local_scope():
            g.ndata['hv'] = (dgl.broadcast_nodes(g, g_feats) + node_feats).view(-1, self.num_head, self.out_feat)
            g.ndata['z'] = (g.ndata['hv'] * self.attn).sum(dim=-1).unsqueeze(dim=-1)

            # ===== 结合度的信息 =====
            g.ndata['z'] = g.ndata['z'] * degree.unsqueeze(dim=1)
            g.ndata['a'] = dgl.softmax_nodes(g, 'z')

            he = dgl.sum_nodes(g, 'hv', 'a')
            # he = F.elu(g_repr)
            he = he.view(-1, self.hidden_dim)
            he = self.lin2(self.lin1(he))
            he = he + g_feats

            return he


class MyReadout(nn.Module):

    def __init__(self, hidden_dim, num_timesteps=2, dropout=0.2):
        super(MyReadout, self).__init__()

        self.readouts = nn.ModuleList()
        for _ in range(num_timesteps):
            self.readouts.append(GlobalPool( hidden_dim, dropout))

    def forward(self, g, node_feats, degree):

        degree = degree * 0.05
        with g.local_scope():
            g.ndata['hv'] = node_feats
            g_feats = dgl.sum_nodes(g, 'hv')

        for readout in self.readouts:
            g_feats = readout(g, node_feats, g_feats, degree)

        return g_feats
