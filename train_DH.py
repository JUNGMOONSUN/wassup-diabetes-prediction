import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
from typing import Optional, List
import numpy as np
import pandas as pd

from preprocess_DH import get_X, get_y
from nn_DH import ANN
from utils_DH import CustomDataset
from tqdm.auto import tqdm
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


def train_one_epoch(
    model: nn.Module,
    criterion: callable,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: str,
) -> float:
    """train one epoch

    Args:
        model: model
        criterion: loss
        optimizer: optimizer
        data_loader: data loader
        device: device
    """
    model.train()
    total_loss = 0.0
    # 추가된 부분: 손실 기록 리스트
    train_losses = []
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        train_losses.append(loss.item())
    return total_loss / len(data_loader.dataset)

    # 추가된 부분: 에포크당 훈련 손실 기록


train_losses_per_epoch = []


# 추가된 부분: 에포크당 훈련 손실을 시각화하기 위한 함수
def plot_training_loss(train_losses_per_epoch):
    plt.plot(train_losses_per_epoch, label="훈련 손실")
    plt.xlabel("에포크")
    plt.ylabel("손실")
    plt.legend()
    plt.show()


def evaluate(
    model: nn.Module,
    criterion: callable,
    data_loader: DataLoader,
    device: str,
    metric: Optional[torchmetrics.metric.Metric] = None,
    multi_metrics: List[torchmetrics.metric.Metric] = None,
) -> float:
    """evaluate

    Args:
        model: model
        criterions: list of criterion functions
        data_loader: data loader
        device: device
    """
    model.eval()
    total_loss = 0.0
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            total_loss += criterion(output, y).item() * len(y)
            if metric is not None:
                metric.update(output, y)
            if multi_metrics is not None:
                for metric in multi_metrics:
                    metric.update(output, y)
    return total_loss / len(data_loader.dataset)


def kfold_cross_validate(
    model: nn.Module,
    criterion: callable,
    device: str,
    X_trn: np.array,
    y_trn: np.array,
    n_splits: int = 5,
):
    from sklearn.model_selection import StratifiedKFold
    from torchmetrics.classification import (
        BinaryAccuracy,
        BinaryF1Score,
        BinaryRecall,
        BinaryPrecision,
    )

    # from torchmetrics import MeanAbsoluteError, MeanSquaredError, MeanSquaredLogError
    from copy import deepcopy

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2023)
    nets = [deepcopy(model) for i in range(n_splits)]
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    """
  classified(분류 - 1,0)
  regressor(회귀-오차값을 구하기 위해),  threshold를 위해 생성
  
  
  # 분류 지표 - 예측값이 맞을 확률
  scores = {
  'accuracy': [], #정확성
  'precision': [], #정도
  'recall': [], #상기하다
  'f1': []
  }
  """

    for i, (trn_idx, val_idx) in enumerate(skf.split(X_trn, y_trn)):
        X, y = torch.tensor(X_trn[trn_idx]), torch.tensor(y_trn[trn_idx])
        X_val, y_val = torch.tensor(X_trn[val_idx]), torch.tensor(y_trn[val_idx])
        ds = CustomDataset(X, y)
        ds_val = CustomDataset(X_val, y_val)
        dl = DataLoader(ds, batch_size=32, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=False)

        net = nets[i].train()

        """
    pbar = tqdm(range(args.epochs))
    for j in pbar:
      mae, mse, msle = MeanAbsoluteError().to(device), MeanSquaredError().to(device), MeanSquaredLogError().to(device)
      optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
      loss = train_one_epoch(net, criterion, optimizer, dl, device)
      loss_val = evaluate(net, criterion, dl_val, device, multi_metrics=[mae, mse, msle])
      mae, mse, msle = mae.compute(), mse.compute(), msle.compute()
      # Get predictions for validation set
      # y_val_pred = net(X_val).detach().numpy()
      pbar.set_postfix(trn_loss=loss, val_loss=loss_val)  # 진행바 우측에 진행상황 표시
    scores["mae"].append(mae.item()) #
    scores["mse"].append(mse.item()) #meansqrt error -> -1, 1 동일하다고 느껴서
    scores["msle"].append(msle.item())
    """

        # 분류(회귀에선 사용X)

        pbar = tqdm(range(args.epochs))
        for j in pbar:
            accuracy, f1, recall, precision = (
                BinaryAccuracy().to(device),
                BinaryF1Score().to(device),
                BinaryRecall().to(device),
                BinaryPrecision().to(device),
            )
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
            loss = train_one_epoch(net, criterion, optimizer, dl, device)

            loss_val = evaluate(
                net,
                criterion,
                dl_val,
                device,
                multi_metrics=[accuracy, f1, recall, precision],
            )
            acc, f1, rec, prec = (
                accuracy.compute().item(),
                f1.compute().item(),
                recall.compute().item(),
                precision.compute().item(),
            )
            scores["accuracy"].append(acc)
            scores["f1"].append(f1)
            scores["recall"].append(rec)
            scores["precision"].append(prec)

            pbar.set_postfix(
                trn_loss=loss, val_loss=loss_val, val_acc=acc
            )  # 진행바 우측에 진행상황 표시

    return scores


def main(args):
    device = torch.device(args.device)

    train_df = pd.read_csv(args.data_train)
    test_df = pd.read_csv(args.data_test)

    # X_trn = get_X(train_df, scaler=args.scaler)
    # y_trn = get_y(train_df)[:,np.newaxis]
    # X_tst = get_X(test_df, scaler=args.scaler)
    X_trn = pd.read_csv("./trn_X.csv").to_numpy(dtype=np.float32)
    y_trn = pd.read_csv("./trn_y.csv").to_numpy(dtype=np.float32)
    X_tst = pd.read_csv("./tst_X.csv").to_numpy(dtype=np.float32)

    ds = CustomDataset(X_trn, y_trn)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=args.shuffle)

    ds_tst = CustomDataset(X_tst)
    dl_tst = DataLoader(ds_tst, batch_size=args.batch_size)

    model = ANN(
        X_trn.shape[-1],
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        use_dropout=args.use_dropout,
        drop_ratio=args.drop_ratio,
    ).to(device)
    print(model)
    loss_func = nn.functional.binary_cross_entropy

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 추가된 부분: 에포크 루프
    for epoch in range(args.epochs):
        # 훈련
        loss = train_one_epoch(model, loss_func, optimizer, dl, device)
        train_losses_per_epoch.append(loss)

        # 훈련 진행 상황 출력
        print(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {loss:.4f}")

    # 추가된 부분: 에포크당 훈련 손실 그래프 플로팅
    plot_training_loss(train_losses_per_epoch)

    scores = kfold_cross_validate(model, loss_func, device, X_trn, y_trn)
    mean_scores = {k: sum(v) / len(v) for k, v in scores.items()}
    print(mean_scores)

    # train with full trainset
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pbar = range(args.epochs)
    if args.pbar:
        pbar = tqdm(pbar)
    for _ in pbar:
        loss = train_one_epoch(model, loss_func, optimizer, dl, device)
        pbar.set_postfix(trn_loss=loss)

    # save pretrained weight
    torch.save(model.state_dict(), args.output)

    # final outuput with testset
    model = ANN(
        X_trn.shape[-1],
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        use_dropout=args.use_dropout,
        drop_ratio=args.drop_ratio,
    ).to(device)
    model.load_state_dict(torch.load(args.output))
    model.eval()

    result = []
    with torch.inference_mode():
        for X in dl_tst:
            X = X[0].to(device)
            output = torch.where(model(X).squeeze() > 0.5, 1, 0).tolist()
            result.extend(output)

    print(result)

    y_tst = pd.read_csv("tst_y.csv").to_numpy(dtype=np.float32)

    cm = confusion_matrix(y_tst, result)
    print(cm)

    # Calculate the accuracy.
    accuracy = metrics.accuracy_score(y_tst, result)

    # Calculate the precision.
    precision = metrics.precision_score(y_tst, result)

    # Calculate the recall.
    recall = metrics.recall_score(y_tst, result)

    # Calculate the F1 score.
    f1_score = metrics.f1_score(y_tst, result)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1_score)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Classification Training", add_help=add_help
    )

    parser.add_argument(
        "--data-train",
        default="./data/train_df.csv",
        type=str,
        help="train dataset path",
    )
    parser.add_argument(
        "--data-test", default="./data/test_df.csv", type=str, help="test dataset path"
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="device (Use cpu/cuda/mps)"
    )
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("--shuffle", default=True, type=bool, help="shuffle")
    parser.add_argument(
        "--epochs",
        default=5,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--pbar", default=True, type=bool, help="progress bar")
    parser.add_argument(
        "-o",
        "--output",
        default="./model.pth",
        type=str,
        help="path to save output model",
    )

    ## model setting
    parser.add_argument(
        "--hidden-dim",
        default=[128, 128, 64, 32],
        type=int,
        nargs="+",
        help="list of dimension of hidden layer",
    )
    parser.add_argument(
        "--activation", default="relu", choices=["sigmoid", "relu", "tanh", "prelu"]
    )
    parser.add_argument("--use_dropout", default=True, type=bool)
    parser.add_argument("--drop_ratio", default=0.3, type=float)

    ## preprocess setting
    parser.add_argument(
        "--scaler", default="standard", choices=["standard", "minmax", "maxabs"]
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
