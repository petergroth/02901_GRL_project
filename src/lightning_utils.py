import numpy as np
import pytorch_lightning as pl
import torch
from torch import optim
from torch_geometric.loader import DataLoader
from torchmetrics.functional import matthews_corrcoef

from src.dataset import CM_graph_dataset, CM_msa_dataset
from src.models import GAT, GCN, MLP


class ProteinRegressor(pl.LightningModule):
    def __init__(self, model_name: str, lr: float, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        if model_name == "GCN":
            self.model = GCN(**model_kwargs)
        elif model_name == "GAT":
            self.model = GAT(**model_kwargs)
        elif model_name == "MLP":
            self.model = MLP(**model_kwargs)

        self.model_name = model_name
        self.lr = lr

    def training_step(self, batch, batch_idx):
        preds = self.model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=preds, target=batch.y
        )
        self.log("train_bce", loss, on_epoch=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.model(batch)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=preds, target=batch.y
        )
        preds = (torch.sigmoid(preds) > 0.5).long()
        target = batch.y.long()

        mcc = matthews_corrcoef(preds, target, num_classes=2)
        self.log("val_bce", bce_loss, on_epoch=True, batch_size=batch.num_graphs)
        self.log("val_mcc", mcc, on_epoch=True, batch_size=batch.num_graphs)
        return mcc, bce_loss

    def test_step(self, batch, batch_idx):
        preds = self.model(batch)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=preds, target=batch.y.float()
        )
        preds = (torch.sigmoid(preds) > 0.5).long()
        target = batch.y.long()
        mcc = matthews_corrcoef(preds, target, num_classes=2)
        self.log("test_bce", bce_loss, on_epoch=True, batch_size=batch.num_graphs)
        self.log("test_mcc", mcc, on_epoch=True, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class CM_datamodule(pl.LightningDataModule):
    def __init__(self, batch_size: int, input_type: str):
        super().__init__()
        self.batch_size = batch_size
        self.input_type = input_type

    def setup(self, stage=None):
        np.random.seed(0)
        if self.input_type == "graph":
            dataset = CM_graph_dataset()
        elif self.input_type == "msa":
            dataset = CM_msa_dataset()
        else:
            raise ValueError

        perm = np.random.permutation(len(dataset))
        n_train = int(len(dataset) * 0.5)
        n_val = int(len(dataset) * 0.25)
        self.train_dataset = dataset[perm[:n_train]]
        self.val_dataset = dataset[perm[n_train : (n_train + n_val)]]
        self.test_dataset = dataset[perm[(n_train + n_val) :]]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
