import torch.nn as nn
import torch.nn.functional as F



class ANN(nn.Module):
  """ 
  def __init__(self, input_dim:int=5, hidden_dim:int=128, dropout:float=0.3, activation=F.sigmoid):
    super().__init__()
    self.lin1 = nn.Linear(input_dim,hidden_dim)
    self.lin2 = nn.Linear(hidden_dim,1)
    self.dropout = nn.Dropout(dropout)
    self.activation = activation
    self.sigmoid = F.sigmoid

  def forward(self, x):
    x = self.lin1(x)
    x = self.activation(x)
    x = self.dropout(x)
    x = self.lin2(x)
    x = self.sigmoid(x) #"이진분류"일 경우 마지막에 꼭! sigmoid #다중분류는 softmax
    return x
   """
  #activation_list = {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU(), "tanh": nn.Tanh(), "prelu": nn.PReLU()}
  
  def __init__(self, input_dim: int=5, hidden_dim: list=[128, 128, 64, 32], activation: str="sigmoid", use_dropout: bool=True, drop_ratio: float=0.3):
    super().__init__()
    dims = [input_dim] + hidden_dim 
    self.dropout = nn.Dropout(drop_ratio)
    self.activation = activation
    #self.activation = activation_list[activation]

    model = [[nn.Linear(dims[i], dims[i+1]), self.dropout if use_dropout else None, self.activation] for i in range(len(dims) - 1)]
    output_layer = [nn.Linear(dims[-1], 1), nn.Sigmoid()] #회귀 sigmoid 제거
    self.module_list= nn.ModuleList(sum(model, []) + output_layer)
  def forward(self, x):
    for layer in self.module_list:
         x = layer(x)
    return x