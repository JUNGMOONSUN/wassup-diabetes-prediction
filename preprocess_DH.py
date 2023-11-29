import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler

########## 수정사항 ###############
# 원하는 feature를 선택할 수 있도록 -> 이미 설계되어 있음
# y가 Survived로 고정되어 있음 -> get_X처럼 이름을 설정하면 자동으로 y를 가져올 수 있도록 설계
# Scaler 설계
##################################


def get_X(df:pd.DataFrame, features:iter=["SEX","AGE_GROUP","BP_LWST","TRIGLYCERIDE","HDL_CHOLE","LDL_CHOLE","HMG","OLIG_PROTE_CD","CREATININE","SGPT_ALT","GAMMA_GTP","SMK_STAT_TYPE_CD","BMI"], scaler= 'standard', scale_columns = None):
    standard = StandardScaler()
    maxabs = MaxAbsScaler()
    minmax = MinMaxScaler()

    # df1 = pd.get_dummies(df[features]).to_numpy(dtype=np.float32)
    df1 = pd.get_dummies(df[features])
    if scale_columns :
        if Scaler == 'standard':
          df1[scale_columns] = standard.fit_transform(df1[scale_columns])
        elif Scaler == 'maxabs':
          df1[scale_columns] = maxabs.fit_transform(df1[scale_columns])
        elif Scaler == 'minmax':
            df1[scale_columns] = minmax.fit_transform(df1[scale_columns])

    return df1.to_numpy(dtype=np.float32)

def get_y(df:pd.DataFrame, feature_name = "D"):
  '''Make the target from a DataFrame.

  Args:
      df: DataFrame
  '''
  return df[feature_name].to_numpy(dtype=np.float32)