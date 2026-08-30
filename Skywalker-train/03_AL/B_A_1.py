# 0.导入工具包
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1.准备数据
data = load_iris()
# 1.1 数据集划分
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=22)

# 2.特征工程/特征预处理
pre = StandardScaler()
x_train = pre.fit_transform(x_train)
x_test = pre.transform(x_test)

# 3.模型 ( 双##代表统计出来具体的以后就不用了，直接输入确认好的效果好的超参数就可以 )
# # 3.1 实例化
# model = KNeighborsClassifier(n_neighbors=3)
# # 3.2 交叉验证网格搜索
# paras_grid = {'n_neighbors': [4, 5, 7, 9]}
# estimator = GridSearchCV(model, param_grid=paras_grid, cv=4)
# estimator.fit(x_train, y_train)
# # 3.3 结果
# print(estimator.best_score_)
# print(estimator.best_estimator_)
# print(estimator.cv_results_)
# 3.4 模型训练、预测 与 评估
model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

print(accuracy_score(y_test, y_predict))

# 如果还有超参数，写法如下所示
# paras_grid = {'n_neighbors': [4, 5, 7, 9], 'lr': [1, 2, 3], 'mo': [0.1, 0.2, 0.3]}
