# 0.导入依赖包
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# 1.读取数据
data = pd.read_csv('手写数字识别.csv')
x = data.iloc[:, 1:]
y = data.iloc[:, 0]


# 2.特征工程/数据预处理
# 2.1 归一化
x = x/255 #python3会自动算成浮点类型，如果写 x = x/255. 则不管如何都是浮点类型
# 2.2 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=22) # stratify是类别，根据类别进行划分

# 7.模型加载
knn = joblib.load('knn.pth')
print(knn.score(x_test, y_test))

# 8.模型预测
img = plt.imread('demo.png')
img = img.reshape(1, -1)/255.
y_predict = knn.predict(img)
print(y_predict)
