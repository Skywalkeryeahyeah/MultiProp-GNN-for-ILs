from sklearn.preprocessing import StandardScaler


def dm02_StandarScaler():
    data = [[90, 2, 10, 40],
            [60, 4, 15, 45],
            [75, 3, 13, 46]]

    # 2.实例化 transformer取什么名字都可以
    transformer = StandardScaler()

    # 3.对原始数据进行变换
    data = transformer.fit_transform(data)

    # 4.打印归一化后的结果
    print(data)

    # 5.打印每一列数据的均值(mean_)和标准差(var_)
    print('transformer.mean -->', transformer.mean_)
    print('transformer.var -->', transformer.var_)

if __name__ == '__main__':
    dm02_StandarScaler()
