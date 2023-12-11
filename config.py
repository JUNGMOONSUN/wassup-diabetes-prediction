import torch
import torch.nn.functional as F
import torchmetrics
from nn import ANN
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryRecall,
    BinaryPrecision,
)

config = {
    "preprocess": {
        "train-csv": "./data/train.csv",
        "test-csv": "./data/test.csv",
        "output-train-feas-csv": "./data/trn_X.csv",
        "output-test-feas-csv": "./data/tst_X.csv",
        "output-train-target-csv": "./data/trn_y.csv",
        "output-test-target-csv": "./data/tst_y.csv",
        "target-col": "D",
    },
    "files": {
        "X_csv": "./data/trn_X.csv",
        "y_csv": "./data/trn_y.csv",
        "Xst_csv": "./data/tst_X.csv",
        "yst_csv": "./data/tst_y.csv",
        "output": "./model.pth",
        "output_csv": "./results/five_fold.csv",
    },
    "model": ANN,
    "model_params": {
        "input_dim": "auto",  # Always will be determined by the data shape
        "hidden_dim": [256, 128, 128, 64],
        "activation": "relu",
        "use_dropout": False,
        "drop_ratio": 0.3,
    },
    "train_params": {
        "data_loader_params": {
            "batch_size": 128,
            "shuffle": True,  # kfold 를 위해 true.... /DataLoader 기본값 true
        },
        "loss": F.binary_cross_entropy,  # 손실함수 설정 필요
        "optim": torch.optim.Adam,
        "optim_params": {
            "lr": 0.001,
        },
        "metric": MetricCollection(
            [BinaryAccuracy(), BinaryF1Score(), BinaryRecall(), BinaryPrecision()]
        ),  # 출력용
        "device": "cuda",
        "epochs": 10,
    },
    "cv_params": {
        "n_split": 5,
    },
}
