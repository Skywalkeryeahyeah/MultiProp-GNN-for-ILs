from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score  # 评估的
import matplotlib.pyplot as plt


def dm04_kmeans():
    # 2.创建数据集，1000个样本，每个样本2个特征，4个质心簇，数据标准差为[0.4, 0.2, 0.2, 0.2]
    x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=22)

    # 绘制原始数据
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], marker='o')
    plt.show()

    # 3.使用KMeans进行聚类，并使用Calinski-Harabasz方法进行评估
    # 3-1 n_clusters=2
    y_pred = KMeans(n_clusters=2, random_state=22, init='k-means++').fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.show()
    print('1->', calinski_harabasz_score(x, y_pred))

    # 3-2 n_clusters=3
    y_pred = KMeans(n_clusters=3, random_state=22).fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.show()
    print('2->', calinski_harabasz_score(x, y_pred))

    # 3-3 n_clusters=4
    y_pred = KMeans(n_clusters=4, random_state=22).fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.show()

    # 4 模型评估
    print('3->', calinski_harabasz_score(x, y_pred))


if __name__ == '__main__':
    dm04_kmeans()


