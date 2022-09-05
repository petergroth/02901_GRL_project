import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, knn_graph, GATv2Conv
from torch_geometric.utils import dropout_adj
from torch_scatter import scatter_mean


class GCN(torch.nn.Module):
    """Simple convolutional GNN"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        dropout: float = 0.0,
        esm: bool = False,
        edge_dropout: float = 0.0,
    ):
        super(GCN, self).__init__()
        self.esm = esm
        self.gcn1 = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        self.gcn2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.gcn3 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.act = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.edge_dropout = edge_dropout

    def forward(self, data):
        if self.esm:
            x, edge_index, batch = data.esm, data.edge_index, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.edge_dropout > 0.0:
            edge_index, _ = dropout_adj(edge_index, p=self.edge_dropout)

        x = self.act(self.dropout(self.gcn1(x=x, edge_index=edge_index)))
        x = self.act(self.dropout(self.gcn2(x=x, edge_index=edge_index)))
        x = self.act(self.dropout(self.gcn3(x=x, edge_index=edge_index)))
        x = self.linear(x)
        x = global_mean_pool(x=x, batch=batch)
        return x.squeeze()


class GAT(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        dropout: float = 0.0,
        heads: int = 4,
        esm: bool = False,
        edge_dropout: float = 0.0,
        edge_dim: int = 0
    ):
        super(GAT, self).__init__()

        self.gat1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim
        )
        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim
        )
        self.gat3 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim
        )
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.act = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.esm = esm
        self.edge_dropout = edge_dropout
        self.use_edges = True if edge_dim > 0 else False

    def forward(self, data):
        x = data.esm if self.esm else data.x
        edge_index, batch = data.edge_index, data.batch
        edge_attr = data.edge_lengths if self.use_edges else None

        if self.edge_dropout > 0.0:
            edge_index, edge_attr = dropout_adj(edge_index, p=self.edge_dropout)

        x = self.act(self.gat1(x=x, edge_index=edge_index, edge_attr=edge_attr))
        x = self.act(self.gat2(x=x, edge_index=edge_index, edge_attr=edge_attr))
        x = self.act(self.gat3(x=x, edge_index=edge_index, edge_attr=edge_attr))
        x = self.linear(x)
        x = global_mean_pool(x=x, batch=batch)
        return x.squeeze()







class MLP(torch.nn.Module):
    """Simple MLP that acts on vector representation of entire graph"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        dropout: float = 0.0,
        esm: bool = False,
    ):
        super(MLP, self).__init__()

        self.esm = esm
        self.mlp = torch.nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.Dropout(p=dropout),
            nn.ELU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Dropout(p=dropout),
            nn.ELU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
        )

    def forward(self, data):
        if self.esm:
            x = scatter_mean(src=data.esm, index=data.batch, dim=0)
        else:
            x = data.x
        x = self.mlp(x)
        return x.squeeze()


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    from src.dataset import CM_graph_dataset

    dataset = CM_graph_dataset()
    esm = True

    model_kwargs = {
        "input_dim": 20 if not esm else 1280,
        "hidden_dim": 128,
        "output_dim": 1,
        "dropout": 0.2,
        "esm": esm,
    }

    # model = GAT(**model_kwargs)
    model = MLP(**model_kwargs)
    loader = DataLoader(dataset, batch_size=10)

    batch = next(iter(loader))

    out = model(batch)
