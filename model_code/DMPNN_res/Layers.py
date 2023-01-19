import torch
import torch.nn as nn

from utilis.function import get_activation_func


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act="ReLU", p_dropout=0.8):
        super(ResidualLayer, self).__init__()
        self.act = get_activation_func(act)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = p_dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # self.reset_parameters()

    # def reset_parameters(self):
    #     glorot_orthogonal(self.lin1.weight, scale=2.0)
    #     self.lin1.bias.data.fill_(0)
    #     glorot_orthogonal(self.lin2.weight, scale=2.0)
    #     self.lin2.bias.data.fill_(0)
    def forward(self, message, message_clone):
        return message + self.dropout_layer(self.act(self.lin2(self.act(self.lin1(message_clone)))))


class DMPNN_Encoder(nn.Module):
    def __init__(self, hidden_dim, radius, p_dropout):
        super(DMPNN_Encoder, self).__init__()

        self.atom_feature_dim = 133
        self.bond_feature_dim = 14
        self.hidden_dim = hidden_dim
        self.bias = False
        self.input_dim = 133 + 14

        self.act_func = get_activation_func("ReLU")

        self.dropout = p_dropout
        self.num_MPNN_layer = radius

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.W_i = nn.Linear(self.input_dim, self.hidden_dim, bias=self.bias)
        self.Res_layer = torch.nn.ModuleList([
            ResidualLayer(self.hidden_dim)
            for _ in range(self.num_MPNN_layer - 1)
        ])
        # self.W_h = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)
        # self.W_h2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)
        self.W_o = nn.Linear(self.atom_feature_dim + self.hidden_dim, self.hidden_dim)

    def forward(self, G, gm):
        dist, angle, torsion, i, j, idx_kj, idx_ji, incomebond_edge_ids, incomebond_index_to_atom = gm

        initial_bonds = torch.cat((G.ndata["atom_feature"][j], G.edata["edge_feature"]),
                                  dim=1)  # fea(j)+fea(j-->i) num_bonds x (133+14)

        inputs = self.W_i(initial_bonds)  # num_bonds x hidden_size
        message = self.act_func(inputs)  # num_bonds x hidden_size
        message_clone = message.clone()

        for layer in self.Res_layer:
            message_clone = message_clone.index_add_(0, idx_ji, message[idx_kj]) - message_clone

            message = layer(message, message_clone)
            # message_ = self.act_func(self.W_h2(self.act_func(self.W_h(message_clone))))
            # message = message + self.dropout_layer(message_)
            # message = self.dropout_layer(message)  # num_bonds x hidden

            message_clone = message.clone()

        message = message_clone

        num_atoms = G.num_nodes()
        atom_message = torch.zeros(num_atoms, self.hidden_dim).to(G.device)
        incomebond_hidden = message[incomebond_edge_ids]
        atom_message = atom_message.index_add_(0, incomebond_index_to_atom, incomebond_hidden)

        a_input = torch.cat([G.ndata["atom_feature"], atom_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        # atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        return atom_hiddens
