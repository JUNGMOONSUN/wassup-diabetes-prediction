import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryRecall, BinaryPrecision
from nn import ANN
from typing import Optional, List
from torchmetrics import MetricCollection
from torch import nn #활성화함수용

config = {

  'files': {
    'X_csv': './trn_X.csv',
    'y_csv': './trn_y.csv',
    'output': './model.pth',
    'output_csv': './results/five_fold.csv',
  },

  'model': ANN,
  'model_params': {
    'input_dim': 'auto', # Always will be determined by the data shape
    'hidden_dim': [128, 128, 64, 32],
    #'hidden_dim': 128,
    #'dropout': 0.3,'
    'drop_ratio': 0.3,
    'activation': nn.ReLU(),
  },

  'train_params': {
    'data_loader_params': {
      'batch_size': 32,
      'shuffle': True,
    },
    'loss': F.binary_cross_entropy, #손실함수 설정 필요
    'optim': torch.optim.Adam,
    'optim_params': {
      'lr': 0.0001,
    },
    #'metric': torchmetrics.MeanSquaredError(squared=False),
    #'metric': BinaryAccuracy(),#에포크당 BinaryAccuracy로 기준을 세워주기 위해?
    'metric': MetricCollection([BinaryAccuracy(), BinaryF1Score(), BinaryRecall(), BinaryPrecision()]), #출력용
    'device': 'cpu',
    'epochs': 3,
  },

  'cv_params':{
    'n_split': 5,
  },

}