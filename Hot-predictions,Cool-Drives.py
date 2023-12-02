import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#저장된 데이터 호출
data = pd.read_csv("./data/use-data.csv")

#호출한 데이터의 속성 확인하기
print(data.head())
print(data.info())
print(data.isna().sum())
print(data.describe())
print(np.shape(data))

# 호출한 데이터의 상관관계를 해석하기 위한 heatmap 작성
corr = data.corr("pearson")
plt.rcParams["figure.figsize"] = (20, 7)
heatmap = sns.heatmap(corr, annot=True, center=0)
heatmap.get_figure().savefig("./result/heatmap.png")

# 사용할 열을 선택하기
index = [3, 4, 5, 6, 9, 10, 11, 8]
df = data.iloc[:, index]
print(df.head(5))
print(df.columns)
print(np.shape(df))

# 사용할 데이터의 이상치 제거하기
for column in df.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 박스플롯 그리기 (이상치 제거 후)
plt.figure(figsize=(10, 6))
plt.title('Boxplot for Each Category (Outliers Removed)')
boxplot = sns.boxplot(data=df)
boxplot.get_figure().savefig("./result/boxplot.png")

# 선형 회귀 모델 학습
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print(Y_pred)

# MSE 계산을 통한 정확도 측정
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error (MSE):", mse)

# 산점도 및 회귀선 시각화
plt.figure(figsize=(10, 6))
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
mse = sns.regplot(x=Y_test, y=Y_pred, scatter_kws={'s': 30, 'alpha':0.5}, line_kws={'color': 'red'})
mse.get_figure().savefig("./result/mse.png")
plt.show()
