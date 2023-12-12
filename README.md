# WASSUP 프로젝트 11조
## 건강보험공단 건강검진데이터를 활용한 당뇨병 예측 모델

오세형
전정현
정문선
정영준
최세준(조장)

## 사용 모델
- XGBClassifier
- RandomForestClassifier
- MLPClassifier
- ANN


## 결과 확인
### XGBClassifier, RandomForestClassifier, MLPClassifier
- exam.ipynb

### ANN_arguparser
```python
python train_DH.py
```

### ANN_config
```python
python preprocess.py
python eval.py
python train.py
```

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


- 5. 타겟인 'D'(=종속변수)와 나머지 피처들이 상관관계를 얼마나 갖는지 파악하기 위해 heatmap 작성
  
  ![image](https://github.com/osh612/wassup-diabetes-prediction/assets/52309060/87058c1f-4a42-4fab-b2eb-6de7667ef47a)


### Feature Engineering
- 종속변수와 상관관계가 아주 낮은 변수들은 학습에 끼치는 영향이 미미할 것이므로 삭제
- 또한 종속변수와 상관관계가 아닌 독립변수들끼리 상관관계 계수가 높을수록 모델 성능에 영향을 끼칠 수 있으므로 변수들끼리 통합하여 새로운 피처를 만들거나 제거
- 예를 들어 히트맵에서 HEIGHT, WEIGHT, WAIST 끼리의 상관관계 계수가 높아서 BMI 지수를 새로 만들고 삭제
- BP_HIGH, BP_LWST 도 서로 상관관계가 높으므로 당뇨와 더 관련이 있는 BP_LWST만 사용
- 콜레스테롤 데이터 중에 TOT를 제외하면 독립변수들끼리 상관관계가 계수가 그닥 높지 않으므로 TOT만 제거
