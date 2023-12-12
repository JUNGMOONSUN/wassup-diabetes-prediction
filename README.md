# WASSUP 프로젝트 11조
## 건강보험공단 건강검진데이터를 활용한 당뇨병 예측 모델

오세형
전정현
정문선
정영준
최세준

## 사용 모델
- XGBClassifier
- RandomForestClassifier
- MLPClassifier
- ANN


## 결과 확인
### XGBClassifier, RandomForestClassifier, MLPClassifier
- exam.ipynb

### ANN_arguparser
- python train_DH.py

### ANN_config
1. python preprocess.py
2. python eval.py
3. python train.py

  ## EDA
### 상황 가정 : 건강검진 데이터를 제공 받았으나 전산사의 오류로 당뇨병을 간단하게 판별할 수 있는 공복 혈당 수치의 데이터가 손실됨
- 이때 나머지 건강검진 데이터를 이용하여 이 사람이 당뇨인지 아닌지를 예측하고 그 확률을 보여주는 모델을 제공하자
- 모델을 학습시킬 땐 공복혈당 수치 데이터가 손실되지 않았던 예전 데이터를 사용한다고 가정


### 우리가 가진 데이터중에서 당뇨를 판정할 수 있는 컬럼은 'BLDS'로 공복혈당 수치를 말한다.
- 따라서 'BLDS'의 수치가 126이 넘어가면 당뇨라고 판단, 새로운 컬럼을 만들어 1을 부여하고 126 미만이면 0으로 처리한다.

- 1. 성별, 나이, 체중 각각 분포
  
  ![image](https://github.com/osh612/wassup-diabetes-prediction/assets/52309060/8aa443c9-7c19-4e9f-808e-e1e13a6eb247)

  ![image](https://github.com/osh612/wassup-diabetes-prediction/assets/52309060/5c366e1a-08dc-47c7-917e-2484a5e3bda6)

- 2. 당뇨인 사람과 아닌 사람의 비율
  
  ![image](https://github.com/osh612/wassup-diabetes-prediction/assets/52309060/3a41190a-b338-494a-8f8b-f2e3f92030a9)

- 3. 당뇨와 콜레스테롤 수치들의 관계 분포
  
  ![image](https://github.com/osh612/wassup-diabetes-prediction/assets/52309060/521c311d-0301-4ec1-b16e-b67b5866446b)

- 4. 당뇨와 간기능검사 수치들의 관계 분포
  
  ![image](https://github.com/osh612/wassup-diabetes-prediction/assets/52309060/23772ab0-669c-429e-94b2-87afbca43d89)
