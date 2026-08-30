# 1.导入依赖包
from sklearn.neighbors import KNeighborsClassifier
import joblib


def KNN():
    # 2.准备数据
    x = [[39, 0, 31],
         [3, 2, 65],
         [2, 3, 55],
         [9, 38, 2],
         [8, 34, 17],
         [5, 2, 57],
         [21, 17, 5],
         [45, 2, 9]]
    y = [0, 1, 2, 2, 2, 1, 0, 0]

    # 3.实例化模型
    estimator = KNeighborsClassifier(n_neighbors=3)  # 表示近邻数量
    print('estimator-->', estimator)

    # 4.模型训练
    estimator.fit(x, y)

    # 5.模型预测
    mypre = estimator.predict([[23, 3, 17]])
    print('mypre-->', mypre)

    # 6.模型保存
    joblib.dump(estimator, './KNN_01.bin')

    # 7.模型加载
    estimator1 = joblib.load('KNN_01.bin')

    # 8.模型预测
    print(estimator1.predict([[90, 86, 6]]))


if __name__ == '__main__':
    KNN()
