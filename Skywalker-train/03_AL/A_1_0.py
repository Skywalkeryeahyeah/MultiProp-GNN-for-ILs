# 0.导入依赖包
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# 1.读取数据
data = pd.read_csv('手写数字识别.csv')
x = data.iloc[:, 1:] # 所有行，第二列到最后一列
y = data.iloc[:, 0] # 所有行，第一列
print(Counter(y))

# 2.显示数据
digit = x.iloc[1].values
img = digit.reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()
