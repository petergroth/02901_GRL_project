import torch
from torchmetrics.functional import matthews_corrcoef

from src.lightning_utils import CM_datamodule


def naive_baseline():
    # Load datamodule
    pass


if __name__ == "__main__":
    dm = CM_datamodule(batch_size=2133, input_type="msa")
    dm.setup()
    # Mean predictor based on training set
    train_loader = dm.train_dataloader()
    train_batch = next(iter(train_loader))
    y_majority = torch.mean(train_batch.y)
    if train_batch.y.sum() > (len(train_batch.y) - train_batch.y.sum()):
        y_majority = 1
    else:
        y_majority = 0
    # Target values
    test_loader = dm.test_dataloader()
    test_batch = next(iter(test_loader))
    y_targets = test_batch.y
    # Predictions
    y_preds = y_majority * torch.ones_like(y_targets)

    # Performance
    test_mcc = matthews_corrcoef(y_preds.long(), y_targets.long(), num_classes=2)
