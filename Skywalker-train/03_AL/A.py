# 0.导入依赖包
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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

# 3.模型训练
# 3.1 实例化
model = KNeighborsClassifier(n_neighbors=3)
# 3.2 训练
model.fit(x_train, y_train)

# 4.模型预测
img = plt.imread('demo.png')
img = img.reshape(1, -1)/255.
y_predict = model.predict(img)
print(y_predict)

y_predict1 = model.predict(x_test)

# 5.模型评估
print(model.score(x_test, y_test))
print(accuracy_score(y_predict1, y_test))

# 6.模型保存
joblib.dump(model, 'knn.pth')

