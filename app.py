import os
import torch
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from torch_geometric.data import Data, Batch

# 引用合并后的模型类
from models_lib import (
    FinalFeaturizer, ILCpModel, ILSolubilityModel, ILSurfaceTensionModel
)

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
featurizer = FinalFeaturizer()

# ==========================================
# 路径修改区：确保这些 .pth 和 .pkl 文件真实存在
# ==========================================
BASE_WEIGHT_PATH = "D:/Desktop/ML_study/IL_Project/weights"

CONFIG = {
    'density': {
        'model_class': ILCpModel,
        'model_path': os.path.join(BASE_WEIGHT_PATH, 'density/best_cp_model.pth'),
        'scaler_y': os.path.join(BASE_WEIGHT_PATH, 'density/sc_y_cp.pkl'),
        'scaler_cond': os.path.join(BASE_WEIGHT_PATH, 'density/sc_cond_cp.pkl'),
        'scaler_p': os.path.join(BASE_WEIGHT_PATH, 'density/sc_p_cp.pkl')
    },
    'heat': {
        'model_class': ILCpModel,
        'model_path': os.path.join(BASE_WEIGHT_PATH, 'heat/best_cp_model.pth'),
        'scaler_y': os.path.join(BASE_WEIGHT_PATH, 'heat/sc_y_cp.pkl'),
        'scaler_cond': os.path.join(BASE_WEIGHT_PATH, 'heat/sc_cond_cp.pkl'),
        'scaler_p': os.path.join(BASE_WEIGHT_PATH, 'heat/sc_p_cp.pkl')
    },
    'solubility': {
        'model_class': ILSolubilityModel,
        'model_path': os.path.join(BASE_WEIGHT_PATH, 'solubility/best_model_co2.pth'),
        'scaler_y': os.path.join(BASE_WEIGHT_PATH, 'solubility/sc_y_co2.pkl'),
        'scaler_cond': os.path.join(BASE_WEIGHT_PATH, 'solubility/sc_cond_co2.pkl'),
        'scaler_p': os.path.join(BASE_WEIGHT_PATH, 'solubility/sc_p_co2.pkl')
    },
    'surface': {
        'model_class': ILSurfaceTensionModel,
        'model_path': os.path.join(BASE_WEIGHT_PATH, 'surface/best_model.pth'),
        'scaler_y': os.path.join(BASE_WEIGHT_PATH, 'surface/sc_target.pkl'),
        'scaler_cond': os.path.join(BASE_WEIGHT_PATH, 'surface/sc_cond.pkl'),
        'scaler_p': os.path.join(BASE_WEIGHT_PATH, 'surface/sc_phys.pkl')
    }
}

# 自动预加载所有模型
models = {}
scalers = {}

print(">>> 正在初始化模型与标准化器...")
for prop, cfg in CONFIG.items():
    if os.path.exists(cfg['model_path']):
        m = cfg['model_class']().to(device)
        m.load_state_dict(torch.load(cfg['model_path'], map_location=device))
        m.eval()
        models[prop] = m
        scalers[prop] = {
            'y': joblib.load(cfg['scaler_y']),
            'cond': joblib.load(cfg['scaler_cond']),
            'p': joblib.load(cfg['scaler_p'])
        }
    else:
        print(f"⚠️ 警告: 找不到 {prop} 的权重文件: {cfg['model_path']}")


# @app.route('/predict', methods=['POST'])
@app.route('/')
def index():
    # 这样访问 http://127.0.0.1:5000 就会看到界面
    return render_template('index.html')

# --- 关键修改：必须加上下面这一行装饰器 ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        prop = data.get('type')
        c_smi = data.get('cation_smiles')
        a_smi = data.get('anion_smiles')
        T = float(data.get('T'))
        P = float(data.get('P', 0.1))
        X = float(data.get('X', 1.0))

        if prop not in models:
            return jsonify({'error': f'Property {prop} not loaded'}), 400

        # 1. 特征工程
        c_res = featurizer.get_features(c_smi)
        a_res = featurizer.get_features(a_smi)
        if not c_res or not a_res:
            return jsonify({'error': 'SMILES 无法解析'}), 400

        # 2. 条件标准化
        cond_vals = [T, X] if prop == 'surface' else [T, P, X]
        cond_n = torch.tensor(scalers[prop]['cond'].transform([cond_vals]), dtype=torch.float).to(device)

        # 3. 物理属性标准化
        p_c = torch.tensor(scalers[prop]['p'].transform([[c_res[3], c_res[4]]]), dtype=torch.float).to(device)
        p_a = torch.tensor(scalers[prop]['p'].transform([[a_res[3], a_res[4]]]), dtype=torch.float).to(device)

        # 4. 图数据转换
        cb = Batch.from_data_list([Data(x=c_res[0], edge_index=c_res[1])]).to(device)
        ab = Batch.from_data_list([Data(x=a_res[0], edge_index=a_res[1])]).to(device)
        cf, af = c_res[2].unsqueeze(0).to(device), a_res[2].unsqueeze(0).to(device)

        # 5. 模型预测
        with torch.no_grad():
            output_scaled = models[prop](cb, ab, cf, af, cond_n, p_c, p_a)
            res = scalers[prop]['y'].inverse_transform(output_scaled.cpu().numpy().reshape(-1, 1))[0][0]

        return jsonify({'status': 'success', 'value': round(float(res), 4)})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    print(">>> IL-GNN 预测服务已启动，监听端口 5000...")
    app.run(debug=True, port=5000)