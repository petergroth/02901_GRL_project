import json

import pandas as pd
import torch
from Bio import SeqIO
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import knn_graph


class CM_graph_dataset(InMemoryDataset):
    def __init__(self, root="data", n_neighbours: int = 10):
        self.n_neighbours = n_neighbours
        self._alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_dict = {aa: i for i, aa in enumerate(self._alphabet)}
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def raw_file_names(self):
        return ["cm.csv", "cm.json"]

    @property
    def processed_file_names(self):
        return ["cm_dataset.pt"]

    def process(self):
        data_list = []
        # Load structures
        with open(self.raw_paths[1], "r") as f:
            data_structure = json.load(f)

        df = pd.read_csv(self.raw_paths[0])
        df["order"] = df["name"].str[7:].astype(int)
        df = df.sort_values(by="order")
        df = df.loc[df["indicator"].isin((0, 1, 2))]
        names = df["name"].tolist()

        # Iterate through proteins in named order. Inefficient, but ensures fixed ordering when comparing
        # to baselines.
        for name in names:
            for protein in data_structure:
                if name == protein["name"]:
                    # Extract positions
                    backbone = torch.tensor(protein["coords"])
                    c_a = backbone[:, 1, :]
                    # KNN graph to define edges
                    edge_index = knn_graph(x=c_a, k=self.n_neighbours)
                    # Normalized edge vectors
                    edge_vector_diffs = c_a[edge_index[1]] - c_a[edge_index[0]]
                    edge_lengths = edge_vector_diffs.norm(dim=1, keepdim=True)
                    edge_vectors = edge_vector_diffs / edge_lengths

                    # One-hot encoding of amino acids
                    one_hot_encoding = torch.zeros((len(backbone), 20))
                    for j, letter in enumerate(protein["sequence"]):
                        if letter in self.aa_dict:
                            k = self.aa_dict[letter]
                            one_hot_encoding[j, k] = 1.0
                    x = one_hot_encoding
                    y = torch.tensor(protein["target_class"]).float()

                    # Load ESM embeddings
                    esm = torch.load(f"data/esm_1b_embeddings/{name}.pt")

                    data = Data(
                        x=x,
                        y=y,
                        edge_index=edge_index,
                        pos=c_a,
                        edge_vectors=edge_vectors,
                        edge_lengths=edge_lengths,
                        esm=esm["representations"][33],
                    )
                    data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class CM_msa_dataset(InMemoryDataset):
    def __init__(self, root="data"):
        self._alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_dict = {aa: i for i, aa in enumerate(self._alphabet)}
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def raw_file_names(self):
        return ["cm.csv", "cm_local.aln.fasta"]

    @property
    def processed_file_names(self):
        return ["cm_msa_dataset.pt"]

    def process(self):
        data_list = []

        # Read data
        df = pd.read_csv(self.raw_paths[0])
        df["order"] = df["name"].str[7:].astype(int)
        df = df.sort_values(by="order")
        df = df.loc[df["indicator"].isin((0, 1, 2))]
        names = df["name"].tolist()

        # Extract MSA length from FASTA
        sequence = next(iter(SeqIO.parse(self.raw_paths[1], "fasta")))
        msa_len = len(sequence.seq)

        # Process each protein sequentially
        for sequence in SeqIO.parse(self.raw_paths[1], "fasta"):
            if sequence.id in names:
                one_hot_encoding = torch.zeros((msa_len, 20))
                for j, letter in enumerate(sequence.seq):
                    if letter in self.aa_dict:
                        k = self.aa_dict[letter]
                        one_hot_encoding[j, k] = 1.0
                x = one_hot_encoding.flatten().unsqueeze(0)
                # Extract target value from df
                y = torch.tensor(
                    df.loc[df["name"] == sequence.id, "target_class"].values
                ).float()
                data = Data(x=x, y=y)
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    d1 = CM_msa_dataset()
    d2 = CM_graph_dataset()
