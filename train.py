import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
import torchmetrics

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix


def train_one_epoch(
    model: nn.Module,
    criterion: callable,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    metric: BinaryAccuracy,
    device: str,
) -> float:
    model.train()
    total_loss = 0
    # count = 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = criterion(output, y)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metric.update(output, y)
        avg_loss = total_loss / len(data_loader)
    return avg_loss


def main(cfg):
    import numpy as np
    import pandas as pd
    from torch.utils.data.dataset import TensorDataset
    from nn import ANN
    from tqdm.auto import trange

    train_params = cfg.get("train_params")
    device = torch.device(train_params.get("device"))

    files = cfg.get("files")
    X_trn = torch.tensor(
        pd.read_csv(files.get("X_csv"), index_col=0).to_numpy(dtype=np.float32)
    )
    y_trn = torch.tensor(
        pd.read_csv(files.get("y_csv"), index_col=0).to_numpy(dtype=np.float32)
    )

    X_tst = torch.tensor(
        pd.read_csv(files.get("Xst_csv"), index_col=0).to_numpy(dtype=np.float32)
    )
    y_tst = torch.tensor(
        pd.read_csv(files.get("yst_csv"), index_col=0).to_numpy(dtype=np.float32)
    )

    dl_params = train_params.get("data_loader_params")
    ds = TensorDataset(X_trn, y_trn)
    dl = DataLoader(ds, **dl_params)  # dataloader는 suffle... 해서 데이터 갖고옴...

    Model = cfg.get("model")
    model_params = cfg.get("model_params")
    model_params["input_dim"] = X_trn.shape[-1]
    model = Model(**model_params).to(device)

    Optim = train_params.get("optim")
    optim_params = train_params.get("optim_params")
    optimizer = Optim(model.parameters(), **optim_params)

    loss = train_params.get("loss")
    metric = train_params.get("metric").to(device)
    values = []
    pbar = trange(train_params.get("epochs"))
    for _ in pbar:
        bar_loss = train_one_epoch(model, loss, optimizer, dl, metric, device)

        values.append(metric.compute())
        metric.reset()
        pbar.set_postfix(loss=bar_loss)

    torch.save(model.state_dict(), files.get("output"))  # 모델 저장
    ##############################################################

    # final outuput with testset
    model = cfg.get("model")
    model_params = cfg.get("model_params")
    model_params["input_dim"] = X_tst.shape[-1]
    model = model(**model_params).to(device)

    resultFile = files.get("output")
    model.load_state_dict(torch.load(resultFile))
    model.eval()

    dst = TensorDataset(X_tst, y_tst)
    dlt_params = train_params.get("data_loader_params")
    dlt_params["batch_size"] = len(dst)
    dlt_params["shuffle"] = False  # suffle = false

    dlt = DataLoader(dst, **dlt_params)

    result = []
    with torch.inference_mode():
        for X in dlt:
            X = X[0].to(device)
            output = torch.where(model(X).squeeze() > 0.5, 1, 0).tolist()
            result.extend(output)

    col_name = ["D"]
    list_df = pd.DataFrame(zip(result), columns=col_name)
    list_df.to_csv("./data/Result.csv", index=False)

    ##########
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
        description="Pytorch K-fold Cross Validation", add_help=add_help
    )
    parser.add_argument(
        "-c", "--config", default="./config.py", type=str, help="configuration file"
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exec(open(args.config).read())
    config = config
    main(config)
