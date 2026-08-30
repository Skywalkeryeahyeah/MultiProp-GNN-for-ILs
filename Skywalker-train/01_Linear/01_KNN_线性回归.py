# 1.导入依赖包
from sklearn.linear_model import LinearRegression
import joblib
# import matplotlib.pyplot as plt


def dm01_Regression_pred():
    # 2.准备数据
    x = [[80, 86], [82, 80], [85, 78], [90, 90], [86, 82], [82, 90], [78, 80], [92, 94]]
    y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

    # 3.实例化 线性回归模型
    estimator = LinearRegression()
    print('estimator-->', estimator)

    # 4.模型训练
    estimator.fit(x, y)
    print('estimator.coef_-->', estimator.coef_)
    print('estimator.intercept_-->', estimator.intercept_)

    # 5.模型预测
    mypred = estimator.predict([[90, 80]])
    print('mypred-->', mypred)


# if __name__ == '__main__':
#     dm01_Regression_pred()

# 这里y=k1x1+k2x2，故estimator.coef_--> [0.3 0.7]，也就是y=0.3x1+0.7x2


    # 6.模型保存
    joblib.dump(estimator, './mymodel01.bin')

    # 7.模型加载
    estimator1 = joblib.load('mymodel01.bin')

    # 8.模型预测
    print(estimator1.predict([[90, 86]]))

    # # R方查看
    # print(f"模型的 R²: {estimator1.score(x, y)}")  # 计算 R²
    #
    # # 绘制回归直线
    # # 选择第一个特征 x1 来绘制散点图
    # x1 = [i[0] for i in x]  # 提取 x 中的第一个特征作为 x 轴
    # plt.scatter(x1, y, color='blue')  # 绘制数据点
    #
    # # 绘制回归直线，基于第一个特征（x1）进行预测
    # plt.plot(x1, estimator1.predict([[i, 86] for i in x1]), color='red')  # 使用预测值来画回归直线
    # plt.xlabel('Feature 1 (x1)')
    # plt.ylabel('Target (y)')
    # plt.title('Linear Regression - Feature 1 vs Target')
    # plt.show()


if __name__ == '__main__':
    dm01_Regression_pred()
    