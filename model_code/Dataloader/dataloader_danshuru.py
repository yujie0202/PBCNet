import dgl
import numpy as np
import pandas as pd
import torch
from utilis.utilis import Extend, pkl_load


def collate_fn(samples):
    ligand1_dir = [s.Ligand1.values[0] for s in samples]
    pocket_dir = [s.Ligand1.values[0].rsplit("/", 1)[0] + "/pocket.pkl" for s in samples]
    graph1_list = [pkl_load(s) for s in ligand1_dir]
    pocket_list = [pkl_load(s) for s in pocket_dir]
    g1 = dgl.batch(graph1_list)
    pock = dgl.batch(pocket_list)
    # index_kj1, index_ji1 = triplets(g1)
    # index_kj2, index_ji2 = triplets(g2)
    label1_list = [s.Lable1.values[0] for s in samples]  # validation samples' labels
    file_name = [s.rsplit("/", 2)[1] for s in ligand1_dir]

    return g1, \
           pock, \
           torch.tensor(label1_list), \
           file_name


class LeadOptDataset():
    def __init__(self, df_path, label_scalar=None):
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)
        self.label_scalar = label_scalar

        if self.label_scalar == "finetune":
            label = self.df.Lable.values
            label = (np.array(label).astype(float) - 0.04191832) / 1.34086546
            self.df["Lable"] = label

        elif self.label_scalar is not None:
            label = self.df.Lable.values
            label = np.reshape(label, (-1, 1))
            self.label_scalar = self.label_scalar.fit(label)
            label = self.label_scalar.transform(label)
            self.df["Lable"] = label.flatten()

        # self.df = self.df[0:256]
        super(LeadOptDataset, self).__init__()

    def file_names_(self):
        ligand_dir = self.df.Ligand1.values
        file_names = [s.rsplit("/", 2)[1] for s in ligand_dir]
        return list(set(file_names))

    def __getitem__(self, idx):
        return self.df[idx:idx + 1]

    def __len__(self):
        return len(self.df)