import os
import torch
import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from torch_geometric.data import Data, Batch

# 请确保 models_lib.py 中包含这四个类名
from models_lib import (
    FinalFeaturizer, ILCpModel, ILSolubilityModel, ILSurfaceTensionModel
)

app = Flask(__name__)
CORS(app)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
featurizer = FinalFeaturizer()

# ================= 1. 权重路径配置 =================
# 建议使用绝对路径或确保运行目录正确
# BASE_WEIGHT_PATH = "D:/Desktop/ML_study/IL_Project/weights"

# ================= 1. 权重路径配置 =================
# 动态获取 app.py 所在的文件夹绝对路径（绝对安全的相对路径写法）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_WEIGHT_PATH = os.path.join(BASE_DIR, 'weights')

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

# ================= 2. 模型预加载逻辑 =================
models = {}
scalers = {}

print(">>> Initializing Models and Scalers...")
for prop, cfg in CONFIG.items():
    try:
        # 加载模型
        m = cfg['model_class']().to(device)
        m.load_state_dict(torch.load(cfg['model_path'], map_location=device, weights_only=True))
        m.eval()
        models[prop] = m

        # 加载标准化器
        scalers[prop] = {
            'y': joblib.load(cfg['scaler_y']),
            'cond': joblib.load(cfg['scaler_cond']),
            'p': joblib.load(cfg['scaler_p'])
        }
        print(f"Successfully loaded: {prop}")
    except Exception as e:
        print(f"Error loading {prop}: {e}")


# ================= 3. 页面路由 =================
@app.route('/')
def index(): return render_template('index.html')


@app.route('/density')
def density_page(): return render_template('density.html')


@app.route('/heat')
def heat_page(): return render_template('heat.html')


@app.route('/solubility')
def solubility_page(): return render_template('solubility.html')


@app.route('/surface')
def surface_page(): return render_template('surface.html')

# 👈 新增这一行：渲染其他模型页面
@app.route('/other')
def other_page(): return render_template('other.html')


# ================= 4. 统一预测接口 =================
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
            return jsonify({'status': 'error', 'message': f'Model {prop} not available'}), 400

        # 特征处理
        c_res = featurizer.get_features(c_smi)
        a_res = featurizer.get_features(a_smi)

        # 数据标准化
        cond_vals = [T, X] if prop == 'surface' else [T, P, X]
        cond_n = torch.tensor(scalers[prop]['cond'].transform([cond_vals]), dtype=torch.float).to(device)
        p_c = torch.tensor(scalers[prop]['p'].transform([[c_res[3], c_res[4]]]), dtype=torch.float).to(device)
        p_a = torch.tensor(scalers[prop]['p'].transform([[a_res[3], a_res[4]]]), dtype=torch.float).to(device)

        # 批处理转换
        cb = Batch.from_data_list([Data(x=c_res[0], edge_index=c_res[1])]).to(device)
        ab = Batch.from_data_list([Data(x=a_res[0], edge_index=a_res[1])]).to(device)
        cf, af = c_res[2].unsqueeze(0).to(device), a_res[2].unsqueeze(0).to(device)

        with torch.no_grad():
            out = models[prop](cb, ab, cf, af, cond_n, p_c, p_a)
            final_res = scalers[prop]['y'].inverse_transform(out.cpu().numpy().reshape(-1, 1))[0][0]

        return jsonify({'status': 'success', 'value': round(float(final_res), 4)})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8000)