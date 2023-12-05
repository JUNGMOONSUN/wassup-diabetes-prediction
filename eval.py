import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryRecall, BinaryPrecision
from dataclasses import dataclass, field
from typing import Type, Optional
import pandas as pd
from torchmetrics import MetricCollection

def evaluate(
  model:nn.Module,
  data_loader:DataLoader,
  metric:torchmetrics.Metric = BinaryAccuracy,
  device:str='cpu',
) -> None:
  '''evaluate
  
  Args:
      model: model
      criterions: list of criterion functions
      data_loader: data loader
      device: device
      metrcis: metrics
  '''
  model.eval()
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      accuracy, f1, recall, precision = BinaryAccuracy().to(device), BinaryF1Score().to(device), BinaryRecall().to(device), BinaryPrecision().to(device)
      output = model(X)
      metric.update(output, y)

@dataclass
class KFoldCV:
  X: torch.Tensor
  y: torch.Tensor
  Model: Type[nn.Module]
  model_args: tuple = tuple()
  model_kwargs: dict = field(default_factory=lambda : {})
  epochs: int = 1
  criterion: callable = F.binary_cross_entropy
  Optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
  optim_kwargs: dict = field(default_factory=lambda : {})
  trn_dl_kwargs: dict = field(default_factory=lambda : {'batch_size': 36})
  val_dl_kwargs: dict = field(default_factory=lambda : {'batch_size': 36})
  n_splits: int = 3
  #metric: torchmetrics.Metric = torchmetrics.MeanSquaredError(squared=False)
  #metric: torchmetrics.Metric = BinaryAccuracy()
  metric: torchmetrics.Metric = MetricCollection([BinaryAccuracy(), BinaryF1Score(), BinaryRecall(), BinaryPrecision()])
  
  device: str = 'cpu'

  def run(self):
    from torch.utils.data import TensorDataset
    #from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    from tqdm.auto import trange
    from train import train_one_epoch

    model = self.Model(*self.model_args, **self.model_kwargs).to(self.device) #모델 생성
    models = [self.Model(*self.model_args, **self.model_kwargs).to(self.device) for _ in range(self.n_splits)] #split 개수만큼 모델 생성
    for m in models:
      m.load_state_dict(model.state_dict()) #가중치초기화
    strFold = StratifiedKFold(n_splits=self.n_splits, shuffle=False)

    metrics = {'BinaryAccuracy':[], 'BinaryF1Score': [], 'BinaryRecall': [], 'BinaryPrecision':[]}
    for i, (trn_idx, val_idx) in enumerate(strFold.split(self.X, self.y)):
      X_trn, y_trn = self.X[trn_idx], self.y[trn_idx]
      X_val, y_val = self.X[val_idx], self.y[val_idx]

      ds_trn = TensorDataset(X_trn, y_trn)
      ds_val = TensorDataset(X_val, y_val)

      dl_trn = DataLoader(ds_trn, **self.trn_dl_kwargs)
      dl_val = DataLoader(ds_val, **self.val_dl_kwargs)

      m = models[i]
      optim = self.Optimizer(m.parameters(), **self.optim_kwargs)
      
      pbar = trange(self.epochs)
      for _ in pbar:
        #Metric Collection
        bar_loss = train_one_epoch(m, self.criterion, optim, dl_trn, self.metric, self.device)
        trn_metric = self.metric.compute()
        self.metric.reset()
        evaluate(m, dl_val, self.metric, self.device)
        val_metric = self.metric.compute()
        self.metric.reset()
        #pbar.set_postfix(val_loss=val_metric)
        pbar.set_postfix(loss=bar_loss)
      
      metrics['BinaryAccuracy'].append(val_metric['BinaryAccuracy'].item())
      metrics['BinaryF1Score'].append(val_metric['BinaryF1Score'].item())
      metrics['BinaryRecall'].append(val_metric['BinaryRecall'].item())
      metrics['BinaryPrecision'].append(val_metric['BinaryPrecision'].item())
      
    return pd.DataFrame(metrics)

def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

  return parser

if __name__ == "__main__":
  import numpy as np
  from nn import ANN

  args = get_args_parser().parse_args()
  
  exec(open(args.config).read())
  cfg = config

  train_params = cfg.get('train_params')
  device = train_params.get('device')

  files = cfg.get('files')
  #X_df = pd.read_csv(files.get('X_csv'), index_col='D')
  #y_df = pd.read_csv(files.get('y_csv'), index_col='D')
  X_df = pd.read_csv(files.get('X_csv'))
  y_df = pd.read_csv(files.get('y_csv'))

  X, y = torch.tensor(X_df.to_numpy(dtype=np.float32)), torch.tensor(y_df.to_numpy(dtype=np.float32))

  Model = cfg.get('model')
  model_params = cfg.get('model_params')
  model_params['input_dim'] = X.shape[-1]
  
  
  dl_params = train_params.get('data_loader_params')

  Optim = train_params.get('optim')
  optim_params = train_params.get('optim_params')

  metric = train_params.get('metric').to(device)
  
  cv = KFoldCV(X, y, Model, model_kwargs=model_params,
               epochs=train_params.get('epochs'),
               criterion=train_params.get('loss'),
               Optimizer=Optim,
               optim_kwargs=optim_params,
               trn_dl_kwargs=dl_params, val_dl_kwargs=dl_params,
               metric=metric,
               device=device)
  res = cv.run()

  res = pd.concat([res])
  print(res)
  res.to_csv(files.get('output_csv'))