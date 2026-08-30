import numpy as np
from sklearn.preprocessing import MinMaxScaler

def dm01_MinMaxscaler():
    data = [[90, 2, 10, 40],
            [60, 4, 15, 45],
            [75, 3, 13, 46]]

    # 2.实例化 transformer取什么名字都可以
    transformer = MinMaxScaler()

    # 3.对原始数据进行变换
    data = transformer.fit_transform(data)

    # 4.打印归一化后的结果
    print(data)

if __name__ == '__main__':
    dm01_MinMaxscaler()