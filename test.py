import requests

# 1. 准备测试数据 (以 [Bmim][BF4] 为例)
test_data = {
    "type": "surface",        # 换成 density, heat 或 solubility 也可以
    "cation_smiles": "CCCCn1cc[n+](C)c1",
    "anion_smiles": "F[B-](F)(F)F",
    "T": 298.15,
    "P": 0.1,
    "X": 1.0
}

# 2. 发送请求到你刚刚启动的 5000 端口
try:
    response = requests.post("http://127.0.0.1:5000/predict", json=test_data)
    print("服务器返回结果:", response.json())
except Exception as e:
    print("连接失败:", e)