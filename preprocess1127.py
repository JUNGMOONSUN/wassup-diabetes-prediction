import pandas as pd
import numpy as np
import torch


def get_X(new_train: pd.DataFrame):
    return torch.tensor(new_train.drop(columns=["BLDS"]).values, dtype=torch.float32)


def g_testX(new_test: pd.DataFrame):
    return torch.tensor(new_test.values, dtype=torch.float32)


def get_y(new_train: pd.DataFrame):
    return torch.tensor(new_train["BLDS"].values, dtype=torch.float32).reshape((-1, 1))
