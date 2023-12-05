import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy

def train_one_epoch(
  model:nn.Module,
  criterion:callable,
  optimizer:torch.optim.Optimizer,
  data_loader:DataLoader,
  metric:BinaryAccuracy,
  device:str
) -> float:
  '''train one epoch
  
  Args:
      model: model
      criterion: loss
      optimizer: optimizer
      data_loader: data loader
      device: device
  '''
  model.train()
  total_loss = 0.
  for X, y in data_loader:
    X, y = X.to(device), y.to(device)
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()#기울기 초기화 하는 부분
    #역전파과정을 통해서 가중치와 편향이 업데이트
    #loss값을 기준으로 기울기계산이 이루어지고 기울기방향에 맞게 기울기 계산이 이루어지면서.. 가중치와 편향 업데이트가 이루어지는겁니다만
    loss.backward()
    optimizer.step()#기울기를 업데이트
    total_loss += loss.item() * len(y)
    metric.update(output, y)#평가지표 - 회귀문제라면 MSE MAE RMSE인데 에폭당 계산

  return total_loss/len(data_loader.dataset)


def main(cfg):
  import numpy as np
  import pandas as pd
  from torch.utils.data.dataset import TensorDataset
  from nn import ANN
  from tqdm.auto import trange
  
  train_params = cfg.get('train_params')
  device = torch.device(train_params.get('device'))
  
  files = cfg.get('files')
  #X_trn = torch.tensor(pd.read_csv(files.get('X_csv'), index_col=0).to_numpy(dtype=np.float32))
  #y_trn = torch.tensor(pd.read_csv(files.get('y_csv'), index_col=0).to_numpy(dtype=np.float32))
  X_trn = torch.tensor(pd.read_csv(files.get('X_csv')).to_numpy(dtype=np.float32))
  y_trn = torch.tensor(pd.read_csv(files.get('y_csv')).to_numpy(dtype=np.float32))

  dl_params = train_params.get('data_loader_params')
  ds = TensorDataset(X_trn, y_trn)
  dl = DataLoader(ds, **dl_params)

  Model = cfg.get('model')
  model_params = cfg.get('model_params')
  model_params['input_dim'] = X_trn.shape[-1]
  model = Model(**model_params).to(device)

  Optim = train_params.get('optim')
  optim_params = train_params.get('optim_params')
  optimizer = Optim(model.parameters(), **optim_params)

  loss = train_params.get('loss')
  metric = train_params.get('metric')
  values = []
  pbar = trange(train_params.get('epochs'))
  for _ in pbar:
    bar_loss = train_one_epoch(model, loss, optimizer, dl, metric, device)
    values.append(metric.compute().items())
    metric.reset()
    pbar.set_postfix(loss=bar_loss)
  torch.save(model.state_dict(), files.get('output'))
  

  
def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  main(config)