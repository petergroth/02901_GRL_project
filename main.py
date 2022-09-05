import seaborn as sns

sns.set_style("dark")

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from src.lightning_utils import CM_datamodule, ProteinRegressor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("setup", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--edge_dropout", type=float, default=0.0)
    args = parser.parse_args()

    # Unpack arguments
    pl.seed_everything(args.seed)
    setup = args.setup
    hidden_dim = args.hidden_dim
    edge_dropout = args.edge_dropout

    # Global parameters
    lr = 3e-4
    batch_size = 64
    output_dim = 1
    dropout = 0.2
    n_epochs = 200
    model_kwargs = {
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "dropout": dropout,
    }

    # Setup specific parameters
    if setup == 0:
        input_type = "msa"
        model_kwargs["esm"] = False
        model_name = "MLP"
        input_dim = 3520
        esm = False

    elif setup == 1:
        input_type = "graph"
        model_name = "MLP"
        input_dim = 1280
        esm = True

    elif setup == 2:
        input_type = "graph"
        model_name = "GAT"
        heads = 4
        model_kwargs["heads"] = heads
        esm = False
        input_dim = 20
        model_kwargs["edge_dropout"] = edge_dropout
        model_kwargs["edge_dim"] = 0

    elif setup == 3:
        input_type = "graph"
        model_name = "GAT"
        heads = 4
        model_kwargs["heads"] = heads
        esm = True
        input_dim = 1280
        model_kwargs["edge_dropout"] = edge_dropout
        model_kwargs["edge_dim"] = 0

    elif setup == 4:
        input_type = "graph"
        model_name = "GCN"
        esm = False
        input_dim = 20
        model_kwargs["edge_dropout"] = edge_dropout

    elif setup == 5:
        input_type = "graph"
        model_name = "GCN"
        esm = True
        input_dim = 1280
        model_kwargs["edge_dropout"] = edge_dropout

    elif setup == 6:
        input_type = "graph"
        model_name = "GAT"
        heads = 4
        model_kwargs["heads"] = heads
        esm = True
        input_dim = 1280
        model_kwargs["edge_dropout"] = edge_dropout
        model_kwargs["edge_dim"] = 1

    else:
        raise ValueError

    model_kwargs["esm"] = esm
    model_kwargs["input_dim"] = input_dim
    dm_kwargs = {"batch_size": batch_size, "input_type": input_type}

    # Instantiations
    regressor = ProteinRegressor(model_name=model_name, lr=lr, **model_kwargs)
    dm = CM_datamodule(**dm_kwargs)
    logger = CSVLogger(save_dir="logs", name=str(setup))
    checkpoint_callback = ModelCheckpoint(
        monitor="val_bce", mode="min", verbose=True, filename="best"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_bce", patience=20, verbose=True, mode="min"
    )
    # Trainer
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
    )
    trainer.fit(model=regressor, datamodule=dm)
    trainer.test(
        model=regressor, datamodule=dm, ckpt_path=checkpoint_callback.best_model_path
    )
