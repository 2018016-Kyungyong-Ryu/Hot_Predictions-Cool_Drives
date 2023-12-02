import pandas as pd
import numpy as np

data = pd.read_csv("./data/use-data.csv")

print(np.shape(data))

ranges_to_drop = (
    list(range(0, 70000)) +
    list(range(100000, 130000)) +
    list(range(200000, 270000)) +
    list(range(300000, 370000)) +
    list(range(400000, 470000)) +
    list(range(500000, 570000)) +
    list(range(600000, 670000)) +
    list(range(700000, 770000)) +
    list(range(800000, 870000)) +
    list(range(900000, 970000))
)

# 데이터 삭제
df = data.drop(labels=ranges_to_drop, axis=0)

print(np.shape(df))
df.to_csv("./data/use-data.csv", index=False)
