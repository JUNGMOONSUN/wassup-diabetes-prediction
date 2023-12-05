from typing import Literal
from dataclasses import dataclass

import pandas as pd

@dataclass
class HealthDataOrigin:
  #file_origin: str = ''
  file_trn: str = './data/train.csv'
  file_tst: str = './data/test.csv'
  index_col: str = ''
  target_col: str = 'D'
  #drop_cols: tuple[str] = ('LotFrontage', 'MasVnrArea', 'GarageYrBlt')

  #fill_num_strategy: Literal['accuracy', 'precision', 'recall','f1'] = 'accuracy'
  
  def _read_origin(self):
      #origin -> origin_df 생성
      origin = pd.read_csv('./data/health_origin.csv')
      #, - 필요
      drop_chole = ['OLIG_PROTE_CD','TOT_CHOLE','TRIGLYCERIDE','HDL_CHOLE','LDL_CHOLE','BLDS']#요단백, 콜레스트롤, 공복혈당

      import numpy as np
      from sklearn.preprocessing import OneHotEncoder      

      #콜레스트롤(대다수가 null) + 추가 null 값
      origin = origin.dropna(subset=drop_chole) 
      
      origin['D'] = origin['BLDS'].apply(lambda x : 1 if x >= 126 else 0)

      origin.to_csv('./data/origin_df.csv')



  def _read_originDf(self):
    import pandas as pd
    origin_df = pd.read_csv('./data/origin_df.csv', index_col=0)
    

    drop_cols = ['BP_HIGH','TOT_CHOLE','DATA_STD_DT','HCHK_YEAR','IDV_ID','SIDO','HEIGHT','WEIGHT','WAIST','SIGHT_LEFT','SIGHT_RIGHT','HEAR_LEFT','HEAR_RIGHT','SGOT_AST','DRK_YN','TTR_YN', 'WSDM_DIS_YN','ODT_TRB_YN','TTH_MSS_YN','CRS_YN', 'HCHK_OE_INSPEC_YN']

    origin_0 = origin_df.loc[origin_df['D'] == 0].head(41767) #언더샘플링
    origin_1 = origin_df.loc[origin_df['D'] == 1]
    origin_df = pd.concat([origin_0, origin_1]) #데이터 합치기
    
    origin_df['BMI'] = origin_df['WEIGHT'] / ((origin_df['HEIGHT'])/100)**2
    origin_df.insert(0, 'BLDS', origin_df.pop('BLDS'))
    
    #drop colunms
    origin_df = origin_df.drop(columns=drop_cols, axis=1)
    
    #suffle
    origin_df = origin_df.sample(frac=1).reset_index(drop=True)

    test_df = origin_df.tail(10000)
    val_df = origin_df.head(73534).tail(10000)
    origin_df = origin_df.head(63534)

    origin_df = origin_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()
    origin_df.to_csv('./data/train.csv', index=False)
    val_df.to_csv('./data/validation.csv', index=False)
    test_df.to_csv('./data/test.csv', index=False)
 


  def _read_df(self, split:Literal['train', 'test']='train'):
    if split == 'train':
      #df = pd.read_csv(self.file_trn, index_col=self.index_col)
      df = pd.read_csv(self.file_trn)
      target = df[self.target_col]
      df = df.drop(self.target_col, axis=1)
      return df, target
    elif split == 'test':
      #df = pd.read_csv(self.file_tst, index_col=self.index_col)
      df = pd.read_csv(self.file_tst)
      target = df[self.target_col]
      df = df.drop(self.target_col, axis=1)
      return df, target #정답 필요
    raise ValueError(f'"{split}" is not acceptable.')


  def preprocess(self):
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    self._read_origin() #1차 EDA process
    self._read_originDf() #2차 EDA process

    trn_df, target = self._read_df('train')
    tst_df, answer = self._read_df('test')

    '''
    # drop `drop_cols`
    trn_df.drop(self.drop_cols, axis=1, inplace=True)
    tst_df.drop(self.drop_cols, axis=1, inplace=True)
    '''
    # Numerical
    trn_num = trn_df.select_dtypes(exclude=['object'])
    tst_num = tst_df.select_dtypes(exclude=['object'])
    # Categorical
    trn_cat = trn_df.select_dtypes(include=['object'])
    tst_cat = tst_df.select_dtypes(include=['object'])

    '''
    # fill the numerical columns using `fill_num_strategy`
    if self.fill_num_strategy == 'mean':
      fill_values = trn_num.mean(axis=1)
    elif self.fill_num_strategy == 'min':
      fill_values = trn_num.min(axis=1)
    elif self.fill_num_strategy == 'max':
      fill_values = trn_num.max(axis=1)
    trn_num.fillna(fill_values, inplace=True)
    tst_num.fillna(fill_values, inplace=True)
    '''
    
    # One-Hot encoding
    #enc = OneHotEncoder(dtype=np.float32, sparse_output=False, drop='if_binary', handle_unknown='ignore')
    enc = OneHotEncoder(dtype=np.float32, sparse=False, drop='if_binary', handle_unknown='ignore')
    trn_cat_onehot = enc.fit_transform(trn_cat)
    
    #error
    tst_cat_onehot = enc.transform(tst_cat)


    trn_arr = np.concatenate([trn_num.to_numpy(), trn_cat_onehot], axis=1)
    tst_arr = np.concatenate([tst_num.to_numpy(), tst_cat_onehot], axis=1)
    trn_X = pd.DataFrame(trn_arr, index=trn_df.index)
    tst_X = pd.DataFrame(tst_arr, index=tst_df.index)

    return trn_X, target, tst_X, answer


def get_args_parser(add_help=True):
  import argparse

  parser = argparse.ArgumentParser(description="Data preprocessing", add_help=add_help)
  # inputs
  parser.add_argument("--train-csv", default="./data/train.csv", type=str, help="train data csv file")
  parser.add_argument("--test-csv", default="./data/test.csv", type=str, help="test data csv file")
  # outputs
  parser.add_argument("--output-train-feas-csv", default="./trn_X.csv", type=str, help="output train features")
  parser.add_argument("--output-test-feas-csv", default="./tst_X.csv", type=str, help="output test features")
  parser.add_argument("--output-train-target-csv", default="./trn_y.csv", type=str, help="output train targets")
  parser.add_argument("--output-test-target-csv", default="./tst_y.csv", type=str, help="output test targets answer") #tst_y.csv(정답지)
  # options
  parser.add_argument("--index-col", default="BLDS", type=str, help="index column")
  parser.add_argument("--target-col", default="D", type=str, help="target column")
  #parser.add_argument("--drop-cols", default=[''], type=list, help="drop columns")
  #parser.add_argument("--fill-num-strategy", default="min", type=str, help="numeric column filling strategy (mean, min, max)")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  health_data = HealthDataOrigin(
    args.train_csv,
    args.test_csv,
    args.index_col,
    args.target_col,
    #args.drop_cols,
    #args.fill_num_strategy
  )
  
  
  trn_X, trn_y, tst_X, tst_y = health_data.preprocess()
  trn_X.to_csv(args.output_train_feas_csv, index=False)
  tst_X.to_csv(args.output_test_feas_csv, index=False)
  trn_y.to_csv(args.output_train_target_csv, index=False)
  tst_y.to_csv(args.output_test_target_csv, index=False)