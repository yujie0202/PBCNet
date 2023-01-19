import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import dgl
import dgl.function as fn
from dgllife.model.readout import AttentiveFPReadout
from ReadoutModel.readout_my import MyReadout
from DMPNN_res.Layers import DMPNN_Encoder
# from EAT.Layer_edge_v import EAT_
# from SIGN.Layers_withoutattention_andangle import SIGN
from Final.final import Bind
from utilis.function import get_activation_func
from dgllife.model.gnn.gcn import GCN
from torch_sparse import SparseTensor
import functools
from dgl.nn.functional import edge_softmax


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


# def atom_cross_att(g1, g2,hidden_dim):
#     t = Variable(torch.tensor(math.sqrt(1 / hidden_dim)), requires_grad=True)
#     batch_size = g1.batch_size

#     for_g1 = []
#     for_g2 = []
#     for i in range(batch_size):
#         hg1 = dgl.slice_batch(g1, i).ndata['h']  # num_atoms * 200
#         hg2 = dgl.slice_batch(g2, i).ndata['h']
#         att = torch.matmul(hg1, torch.transpose(hg2, -1, -2)) * t  # num atoms A * num atoms B
#         att_2 = nn.functional.softmax(att, dim=-1)
#         att_1 = nn.functional.softmax(att, dim=-2)
#         for_hg1 = torch.matmul(att_2, hg2)  # num atoms A * 200
#         for_hg2 = torch.matmul(torch.transpose(att_1, -1, -2), hg1)  # num atoms B * 200
#         for_g1.append(for_hg1)
#         for_g2.append(for_hg2)

#     h1 = torch.cat(for_g1, dim=0)
#     h2 = torch.cat(for_g2, dim=0)

#     return h1, h2


class Cross_Att(nn.Module):
    def __init__(self,hidden_dim):
        super(Cross_Att, self).__init__()

        # self.hidden_dim = hidden_dim
        # self.num_head = num_head
        # self.out_feat = int(hidden_dim / num_head)

        self.k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.t = Variable(torch.tensor(math.sqrt(1 / hidden_dim)), requires_grad=True)
        # self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_head, self.out_feat)))

    def forward(self,g1, g2):
        with g1.local_scope():
            with g2.local_scope():
                batch_size = g1.batch_size

                g1.ndata['k'] = self.k(g1.ndata['h'])
                g2.ndata['k'] = self.k(g2.ndata['h'])

                g1.ndata['q'] = self.q(g1.ndata['h'])
                g2.ndata['q'] = self.q(g2.ndata['h'])

                for_g1 = []
                for_g2 = []

                for i in range(batch_size):
                    k1, q1 = dgl.slice_batch(g1, i).ndata['k'], dgl.slice_batch(g1, i).ndata['q']  # num_atoms * 200
                    k2, q2 = dgl.slice_batch(g2, i).ndata['k'], dgl.slice_batch(g2, i).ndata['q']  # num_atoms * 200

                    att_for_g1 = torch.matmul(q1, torch.transpose(k2, -1, -2)) * self.t  # 更新g1中原子时使用的ATT    num atoms A * num atoms B
                    att_for_g2 = torch.matmul(q2, torch.transpose(k1, -1, -2)) * self.t  # 更新g2中原子时使用的ATT    num atoms B * num atoms A

                    att_for_g1 = nn.functional.softmax(att_for_g1, dim=-1)
                    att_for_g2 = nn.functional.softmax(att_for_g2, dim=-1)

                    for_hg1 = torch.matmul(att_for_g1, k2)  # num atoms A * 200
                    for_hg2 = torch.matmul(att_for_g2, k1)  # num atoms B * 200

                    for_g1.append(for_hg1)
                    for_g2.append(for_hg2)

                h1 = torch.cat(for_g1, dim=0)
                h2 = torch.cat(for_g2, dim=0)

                return h1, h2

# class Cross_Att(nn.Module):
#     def __init__(self, hidden_dim, num_head, feat_drop=0, attn_drop=0, activation="ReLU", activation_att="LeakyReLU"):
#         super(Cross_Att, self).__init__()

#         self.feat_drop = nn.Dropout(feat_drop)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.out_feat = int(hidden_dim / num_head)
#         self.num_head = num_head
#         self.hidden_dim = hidden_dim

#         self.k = nn.Linear(hidden_dim, hidden_dim, bias=True)
#         self.q = nn.Linear(hidden_dim, hidden_dim, bias=True)
#         self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_head, self.out_feat)))
#         self.act_att = get_activation_func(activation_att)

#     def forward(self,g1, g2):
#         with g1.local_scope():
#             with g2.local_scope():

#                 node1 = g1.batch_num_nodes()
#                 node2 = g2.batch_num_nodes()

#                 g1.ndata['from'] = torch.tensor([1]*g1.num_nodes()).to(g1.device)
#                 g2.ndata['from'] = torch.tensor([2]*g2.num_nodes()).to(g1.device)

#                 # e_keys = list(g1.edata.keys())
#                 # n_keys = list(g1.ndata.keys())
#                 # n_keys.remove('h')
#                 # n_keys.remove('from')

#                 # for e_key in e_keys:
#                 #     g1.edata.pop(e_key)
#                 #     g2.edata.pop(e_key)
#                 # for n_key in n_keys:
#                 #     g1.ndata.pop(n_key)
#                 #     g2.ndata.pop(n_key)

#                 G1 = dgl.unbatch(g1)
#                 G2 = dgl.unbatch(g2)
                
#                 # A与B组合，并删除现有的bond
#                 gx = [ dgl.remove_edges(dgl.batch([G1[i],G2[i]]), eids=range(dgl.batch([G1[i], G2[i]]).num_edges())) for i in range(len(G1))]
#                 # 添加新的bond, 前者为[0,0,0,1,1,1], 后者为[0,1,2,0,1,2]
#                 gx = dgl.batch([ dgl.add_edges(gx[i],[j for j in range(node1[i]) for a in range(node2[i])],[j for a in range(node1[i]) for j in range(node1[i],node2[i]+node1[i])]) for i in range(len(gx))])

#                 gx.ndata['k'] = self.k(gx.ndata['h']).view(-1, self.num_head, self.out_feat)
#                 gx.ndata['q'] = self.q(gx.ndata['h']).view(-1, self.num_head, self.out_feat)

#                 gx.apply_edges(fn.u_add_v('k', 'q', 'e'))
#                 att = self.act_att(gx.edata.pop('e'))  # (num_src_edge, num_heads, out_dim)
#                 att = (att * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # (num_edge, num_heads, 1)

#                 att = edge_softmax(gx, att)
#                 gx.edata['att'] = self.attn_drop(att)

#                 gx.update_all(dgl.function.u_mul_e('k', 'att', 'm'), fn.sum('m', 'ft'))   # 这里是用 k 还是用h可以消融一下

#                 h1 = gx.ndata['ft'].view(-1, self.hidden_dim)[gx.ndata['from'] == 1]
#                 h2 = gx.ndata['ft'].view(-1, self.hidden_dim)[gx.ndata['from'] == 2]

#                 print(h1.shape)
#                 print(h2.shape)

#                 return h1, h2


# def _rbf(D, D_min=0., D_max=5., D_count=16, device='cpu'):
#     '''
#     From https://github.com/jingraham/neurips19-graph-protein-design
#
#     Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
#     That is, if `D` has shape [...dims], then the returned tensor will have
#     shape [...dims, D_count].
#     '''
#     D_mu = torch.linspace(D_min, D_max, D_count, device=device)
#     D_mu = D_mu.view([1, -1])
#     D_sigma = (D_max - D_min) / D_count
#     D_expand = torch.unsqueeze(D, -1)
#
#     RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
#     return RBF




def distance(edge_feat):
    def func(edges):
        return {'dist': (edges.src[edge_feat]-edges.dst[edge_feat]).pow(2).sum(dim=-1).sqrt()}

    return func


# def distance_emb(edge_feat, device):
#     def func(edges):
#         return {'rbf': _rbf(edges.data[edge_feat],device=device)}
#
#     return func

def edge_cat(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: torch.cat([edges.data[src_field], edges.data[dst_field]],dim=-1)}

    return func


class DMPNN(nn.Module):
    def __init__(self, hidden_dim, radius, T, p_dropout, ffn_num_layers, num_heads=4,
                 in_dim_atom=70, in_dim_edge=14, encoder_type="DMPNN_res", readout_type="AttFP",output_dim=1, degree_information=0, GCN_=0,
                 cs=0, two_task = 0):
        super(DMPNN, self).__init__()

        self.encoder_type = encoder_type
        self.readout_type = readout_type
        self.degree_information = degree_information
        self.GCN_ = GCN_
        self.cs = cs
        # self.mu = nn.Parameter(torch.tensor([2.0], dtype=torch.float32), requires_grad=False)
        # self.v = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=False)

        self.Cross_Att = Cross_Att(hidden_dim)


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

        ffn = [nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.ReLU()]
        ffn.append(nn.Linear(hidden_dim * 2, hidden_dim))
        ffn.append(nn.ReLU())
        ffn.append(nn.Linear(hidden_dim, int(hidden_dim * 0.5)))
        ffn.append(nn.ReLU())
        ffn.append(nn.Linear(int(hidden_dim * 0.5), self.output_dim))

        self.FNN = nn.Sequential(*ffn)

        # ffn = [nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.ReLU()]
        # self.FNN = nn.Sequential(*ffn)

        # ffn_1 = []
        # ffn_1.append(nn.Linear(hidden_dim * 2, hidden_dim))
        # ffn_1.append(nn.ReLU())
        # ffn_1.append(nn.Linear(hidden_dim, int(hidden_dim * 0.5)))
        # ffn_1.append(nn.ReLU())
        # ffn_1.append(nn.Linear(int(hidden_dim * 0.5), self.output_dim))

        # self.FNN_1 = nn.Sequential(*ffn_1)






        self.GCN = GCN(in_feats=70, hidden_feats=[hidden_dim, hidden_dim, hidden_dim],
                       batchnorm=[True, True, True])

        self.lin_atom1 = DenseLayer(in_dim=in_dim_atom, out_dim=hidden_dim, bias=True)
        self.lin_edge1 = DenseLayer(in_dim=in_dim_edge, out_dim=int(hidden_dim / 2), bias=True)
        # self.lin_rbf = DenseLayer(in_dim=16, out_dim=int(hidden_dim / 2))

        # self.ln = nn.LayerNorm(hidden_dim)
        # self.degree_encoder = nn.Embedding(200, hidden_dim, padding_idx=0)

        # cross attention
        # self.down_lin1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.down_lin2 = nn.Linear(hidden_dim, hidden_dim, bias=False)


        self.lin1_cs = DenseLayer(hidden_dim, hidden_dim)
        self.lin2_cs = DenseLayer(hidden_dim, hidden_dim)

        if two_task == 1:

            ffn_2 = [nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.ReLU()]
            # ffn_2 = []
            ffn_2.append(nn.Linear(hidden_dim * 2, hidden_dim))
            ffn_2.append(nn.ReLU())
            ffn_2.append(nn.Linear(hidden_dim, int(hidden_dim * 0.5)))
            ffn_2.append(nn.ReLU())
            ffn_2.append(nn.Linear(int(hidden_dim * 0.5),  2))   # high or low

            self.FNN2 = nn.Sequential(*ffn_2)

        self.two_task = two_task





    def triplets(self, g):
        row, col =  g.edges()  # j --> i
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

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    # def mu_and_v(self):
    #     self.mu.requires_grad_()
    #     self.v.requires_grad_()

    def forward(self, g1, g2, g_pocket=None):

        i1, j1, idx_i1, idx_j1, idx_k1, idx_kj1, idx_ji1 = self.triplets(g1)
        i2, j2, idx_i2, idx_j2, idx_k2, idx_kj2, idx_ji2 = self.triplets(g2)

        k1 = torch.nonzero(g1.ndata["p_or_l"], as_tuple=True)[0]  # 口袋原子索引
        k2 = torch.nonzero(g2.ndata["p_or_l"], as_tuple=True)[0]

        k1_ = torch.nonzero(g1.ndata["p_or_l"] == 0, as_tuple=True)[0]  # 配体原子索引
        k2_ = torch.nonzero(g2.ndata["p_or_l"] == 0, as_tuple=True)[0]

        if self.GCN_ == 1:   # 打开
            g_pocket = dgl.add_self_loop(g_pocket)
            g_pocket.ndata["atom_feature_h"] = self.GCN(g_pocket, g_pocket.ndata["atom_feature"].to(torch.float32))

            g1.ndata["atom_feature_h"] = torch.zeros([g1.ndata["atom_feature"].shape[0], self.hidden_dim]).to(device=g1.device)
            g2.ndata["atom_feature_h"] = torch.zeros([g2.ndata["atom_feature"].shape[0], self.hidden_dim]).to(device=g1.device)

            g1.ndata["atom_feature_h"][k1_] = self.lin_atom1(
                g1.ndata["atom_feature"].to(torch.float32)[k1_])  # 70 --> 200
            g2.ndata["atom_feature_h"][k2_] = self.lin_atom1(g2.ndata["atom_feature"].to(torch.float32)[k2_])

            g1.ndata["atom_feature_h"][k1] = g_pocket.ndata["atom_feature_h"]  # 卷积之后的口袋信息更新现有信息
            g2.ndata["atom_feature_h"][k2] = g_pocket.ndata["atom_feature_h"]

        if self.GCN_ == 0:   # 关闭
            g1.ndata["atom_feature_h"] = self.lin_atom1(g1.ndata["atom_feature"].to(torch.float32))  # 70 --> 200
            g2.ndata["atom_feature_h"] = self.lin_atom1(g2.ndata["atom_feature"].to(torch.float32))

        if self.degree_information == 1:
            g1.ndata["atom_feature_h"] = g1.ndata["atom_feature_h"] + self.degree_encoder(g1.in_degrees())  # 度信息
            g2.ndata["atom_feature_h"] = g2.ndata["atom_feature_h"] + self.degree_encoder(g2.in_degrees())

        g1.edata["edge_feature_h"] = self.lin_edge1(g1.edata["edge_feature"].to(torch.float32))  # 14 --> 100
        g2.edata["edge_feature_h"] = self.lin_edge1(g2.edata["edge_feature"].to(torch.float32))

        g1.apply_edges(distance('atom_coordinate'))  # 获得dist
        g2.apply_edges(distance('atom_coordinate'))

        # 取log并*2
        diss1 = torch.where(g1.edata['attention_weight'] == 0,
                            torch.tensor(-1).to(device=g1.device, dtype=torch.float32),
                            torch.log(g1.edata['attention_weight'])*2)

        diss2 = torch.where(g2.edata['attention_weight'] == 0,
                            torch.tensor(-1).to(device=g2.device, dtype=torch.float32),
                            torch.log(g2.edata['attention_weight'])*2)

        # 共价边恢复成1
        g1.edata['dist_decay'] = torch.where(g1.edata['attention_weight'] == 1,
                                             torch.tensor(1.0).to(device=g1.device, dtype=torch.float32),
                                             diss1)
        g2.edata['dist_decay'] = torch.where(g2.edata['attention_weight'] == 1,
                                             torch.tensor(1.0).to(device=g1.device, dtype=torch.float32),
                                             diss2)

        # g1.edata['dist_decay'] = torch.where(g1.edata["edge_feature"][:, 0] == 1,
        #                                      torch.tensor(1.0).to(device=g1.device, dtype=torch.float32),
        #                                      torch.tensor(0).to(device=g1.device, dtype=torch.float32))
        # g2.edata['dist_decay'] = torch.where(g2.edata["edge_feature"][:, 0] == 1,
        #                                      torch.tensor(1.0).to(device=g1.device, dtype=torch.float32),
        #                                      torch.tensor(0).to(device=g1.device, dtype=torch.float32))
        # noncon1_ = torch.nonzero(g1.edata["dist_decay"] == 0, as_tuple=True)[0]  # 非共价边索引
        # noncon2_ = torch.nonzero(g2.edata["dist_decay"] == 0, as_tuple=True)[0]

        # # 非共价键按照距离划分
        # g1.edata['dist_decay'][noncon1_] = torch.where(g1.edata["dist"][noncon1_] > self.mu,
        #                                    torch.exp(-torch.pow(g1.edata['dist'][noncon1_] - self.mu, 2) / (self.v + 1e-6)).to(
        #                                          torch.float32),
        #                                    torch.tensor(1.0).to(device=g1.device, dtype=torch.float32))
        # g2.edata['dist_decay'][noncon2_] = torch.where(g2.edata['dist'][noncon2_] > self.mu,
        #                                    torch.exp(-torch.pow(g2.edata['dist'][noncon2_] - self.mu, 2) / (self.v + 1e-6)).to(
        #                                          torch.float32),
        #                                    torch.tensor(1.0).to(device=g2.device, dtype=torch.float32))

        # g1.edata['dist_decay'] = g1.edata['dist_decay'].unsqueeze(-1)
        # g2.edata['dist_decay'] = g2.edata['dist_decay'].unsqueeze(-1)

        h1,att1 = self.encoder(g1, idx_kj1, idx_ji1, idx_i1, idx_j1, idx_k1)
        h2,att2 = self.encoder(g2, idx_kj2, idx_ji2, idx_i2, idx_j2, idx_k2)


        with g1.local_scope():
            with g2.local_scope():
                g1.ndata['h'] = h1
                g2.ndata['h'] = h2

                d1 = g1.in_degrees()[k1_].unsqueeze(dim=-1)
                d2 = g2.in_degrees()[k2_].unsqueeze(dim=-1)

                g1.remove_nodes(k1)
                g2.remove_nodes(k2)

                if self.cs == 1:
                    # atom cross docking
                    # h1_ca, h2_ca = atom_cross_att(g1, g2, self.hidden_dim)

                    h1_ca, h2_ca = self.Cross_Att(g1, g2)

                    # 9.7 mol sub
                    g1.ndata['cs_h'] = h1_ca
                    g2.ndata['cs_h'] = h2_ca
                    # hsg1_cs = self.readout(g1, g1.ndata['cs_h'], False)
                    # hsg2_cs = self.readout(g2, g2.ndata['cs_h'], False)



                    h1_ca = self.lin2_cs(self.lin1_cs(h1_ca))
                    h2_ca = self.lin2_cs(self.lin1_cs(h2_ca))

                    # g1.ndata['h'] = self.act_func(self.down_lin1(torch.cat([g1.ndata['h'], h1_ca], dim=-1)))
                    # g2.ndata['h'] = self.act_func(self.down_lin1(torch.cat([g2.ndata['h'], h2_ca], dim=-1)))

                    g1.ndata['h'] = torch.sub(g1.ndata['h'], h1_ca)
                    g2.ndata['h'] = torch.sub(g2.ndata['h'], h2_ca)

                # g1.ndata['h'] = g1.ndata['h'] + self.degree_encoder(d1)  # 度信息
                # g2.ndata['h'] = g2.ndata['h'] + self.degree_encoder(d2)

                if self.readout_type == "AttFP":
                    hsg1 = self.readout(g1, g1.ndata['h'], False)
                    hsg2 = self.readout(g2, g2.ndata['h'], False)

                elif self.readout_type == "My":
                    hsg1 = self.readout(g1, g1.ndata['h'], d1)
                    hsg2 = self.readout(g2, g2.ndata['h'], d2)

                elif self.readout_type == "mean":
                    hsg1 = dgl.readout_nodes(g1, 'h',op='mean')
                    hsg2 = dgl.readout_nodes(g2, 'h',op='mean')
  
                elif self.readout_type == "sum":
                    hsg1 = dgl.readout_nodes(g1, 'h',op='sum')
                    hsg2 = dgl.readout_nodes(g2, 'h',op='sum')

                # hsg1 = torch.cat([hsg1, dgl.readout_nodes(g1, 'interaction_embedding', op='mean')], dim=-1)  # 非键相互作用力信息
                # hsg2 = torch.cat([hsg2, dgl.readout_nodes(g2, 'interaction_embedding', op='mean')], dim=-1)

                # if self.cs == 1:
                #     hsg1_cs_ = self.lin2_cs(self.lin1_cs(torch.cat([hsg1, hsg1_cs], dim=-1)))
                #     hsg2_cs_ = self.lin2_cs(self.lin1_cs(torch.cat([hsg2, hsg2_cs], dim=-1)))

                #     hsg1 = hsg1 - hsg1_cs_
                #     hsg2 = hsg2 - hsg2_cs_

                zk1 = self.FNN(
                    torch.cat([hsg1.to(torch.float32), hsg2.to(torch.float32), (hsg1 - hsg2).to(torch.float32)],
                              dim=-1))
                # zk1 = self.FNN_1(zk)

                # zk = self.FNN(
                #     torch.cat([hsg1.to(torch.float32), hsg2.to(torch.float32), (hsg1 - hsg2).to(torch.float32), ((hsg1_cs+hsg2_cs)/2).to(torch.float32)],
                #               dim=-1))
                # zk = self.FNN(
                #     torch.cat([hsg1.to(torch.float32), hsg2.to(torch.float32), hsg1_cs.to(torch.float32), hsg2_cs.to(torch.float32)],
                #               dim=-1))

                if self.two_task == 1:
                    zk2 = self.FNN2(torch.cat([hsg1.to(torch.float32), hsg2.to(torch.float32), (hsg1 - hsg2).to(torch.float32)],
                              dim=-1))

                    # zk2 = self.FNN_2(zk)
                    return zk1,zk2,att1,att2

                else:
                    return zk1,att1,att2

