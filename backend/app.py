import os
import torch
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS

from torch_geometric.data import Data, Batch


from backend.models_lib import (
    FinalFeaturizer,
    ILCpModel,
    ILSolubilityModel,
    ILSurfaceTensionModel
)


app = Flask(__name__)

# 允许 React 前端访问
CORS(app)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


featurizer = FinalFeaturizer()



# ==========================================
# 自动定位项目根目录
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


BASE_WEIGHT_PATH = os.path.join(
    BASE_DIR,
    "weights"
)



# ==========================================
# 模型配置
# ==========================================

CONFIG = {


    "density": {

        "model_class": ILCpModel,

        "model_path": os.path.join(
            BASE_WEIGHT_PATH,
            "density",
            "best_cp_model.pth"
        ),

        "scaler_y": os.path.join(
            BASE_WEIGHT_PATH,
            "density",
            "sc_y_cp.pkl"
        ),

        "scaler_cond": os.path.join(
            BASE_WEIGHT_PATH,
            "density",
            "sc_cond_cp.pkl"
        ),

        "scaler_p": os.path.join(
            BASE_WEIGHT_PATH,
            "density",
            "sc_p_cp.pkl"
        )
    },



    "heat": {

        "model_class": ILCpModel,

        "model_path": os.path.join(
            BASE_WEIGHT_PATH,
            "heat",
            "best_cp_model.pth"
        ),

        "scaler_y": os.path.join(
            BASE_WEIGHT_PATH,
            "heat",
            "sc_y_cp.pkl"
        ),

        "scaler_cond": os.path.join(
            BASE_WEIGHT_PATH,
            "heat",
            "sc_cond_cp.pkl"
        ),

        "scaler_p": os.path.join(
            BASE_WEIGHT_PATH,
            "heat",
            "sc_p_cp.pkl"
        )
    },



    "surface": {

        "model_class": ILSurfaceTensionModel,

        "model_path": os.path.join(
            BASE_WEIGHT_PATH,
            "surface",
            "best_model.pth"
        ),

        "scaler_y": os.path.join(
            BASE_WEIGHT_PATH,
            "surface",
            "sc_target.pkl"
        ),

        "scaler_cond": os.path.join(
            BASE_WEIGHT_PATH,
            "surface",
            "sc_cond.pkl"
        ),

        "scaler_p": os.path.join(
            BASE_WEIGHT_PATH,
            "surface",
            "sc_phys.pkl"
        )
    },



    "solubility": {

        "model_class": ILSolubilityModel,

        "model_path": os.path.join(
            BASE_WEIGHT_PATH,
            "solubility",
            "best_model_co2.pth"
        ),

        "scaler_y": os.path.join(
            BASE_WEIGHT_PATH,
            "solubility",
            "sc_y_co2.pkl"
        ),

        "scaler_cond": os.path.join(
            BASE_WEIGHT_PATH,
            "solubility",
            "sc_cond_co2.pkl"
        ),

        "scaler_p": os.path.join(
            BASE_WEIGHT_PATH,
            "solubility",
            "sc_p_co2.pkl"
        )
    }

}



# ==========================================
# 加载模型
# ==========================================

models = {}

scalers = {}


print(">>> 正在初始化模型与标准化器...")


for prop, cfg in CONFIG.items():


    if not os.path.exists(
        cfg["model_path"]
    ):

        print(
            f"⚠️ 未找到 {prop} 权重:"
            f"{cfg['model_path']}"
        )

        continue



    model = cfg["model_class"]().to(device)


    model.load_state_dict(

        torch.load(
            cfg["model_path"],
            map_location=device
        )

    )


    model.eval()


    models[prop] = model



    scalers[prop] = {

        "y":
            joblib.load(
                cfg["scaler_y"]
            ),


        "cond":
            joblib.load(
                cfg["scaler_cond"]
            ),


        "p":
            joblib.load(
                cfg["scaler_p"]
            )

    }


    print(
        f">>> {prop} 模型加载成功"
    )





# ==========================================
# 测试接口
# ==========================================

@app.route("/")
def index():

    return jsonify({

        "status": "running",

        "message":
        "MultiProp-GNN Backend"

    })





# ==========================================
# 预测接口
# ==========================================


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():


    try:


        data = request.json


        prop = data.get(
            "type"
        )


        c_smi = data.get(
            "cation_smiles"
        )


        a_smi = data.get(
            "anion_smiles"
        )


        T = float(
            data.get("T")
        )


        P = float(
            data.get(
                "P",
                0.1
            )
        )


        X = float(
            data.get(
                "X",
                1.0
            )
        )



        if prop not in models:


            return jsonify({

                "status":
                "error",

                "message":
                f"{prop} model not loaded"

            }),400




        # ======================
        # RDKit 特征
        # ======================


        c_res = featurizer.get_features(
            c_smi
        )


        a_res = featurizer.get_features(
            a_smi
        )



        if c_res is None or a_res is None:


            return jsonify({

                "status":
                "error",

                "message":
                "SMILES解析失败"

            }),400





        # ======================
        # 条件特征
        # ======================


        if prop == "surface":

            cond = [T, X]

        else:

            cond = [T, P, X]



        cond_n = torch.tensor(

            scalers[prop]["cond"]
            .transform(
                [cond]
            ),

            dtype=torch.float

        ).to(device)





        # ======================
        # 分子物理描述符
        # ======================


        p_c = torch.tensor(

            scalers[prop]["p"]
            .transform(

                [[
                    c_res[3],
                    c_res[4]
                ]]

            ),

            dtype=torch.float

        ).to(device)



        p_a = torch.tensor(

            scalers[prop]["p"]
            .transform(

                [[
                    a_res[3],
                    a_res[4]
                ]]

            ),

            dtype=torch.float

        ).to(device)





        # ======================
        # 图数据
        # ======================


        cb = Batch.from_data_list(

            [

                Data(
                    x=c_res[0],
                    edge_index=c_res[1]
                )

            ]

        ).to(device)



        ab = Batch.from_data_list(

            [

                Data(
                    x=a_res[0],
                    edge_index=a_res[1]
                )

            ]

        ).to(device)




        c_fp = c_res[2].unsqueeze(0).to(device)

        a_fp = a_res[2].unsqueeze(0).to(device)





        # ======================
        # 推理
        # ======================


        with torch.no_grad():


            pred = models[prop](

                cb,

                ab,

                c_fp,

                a_fp,

                cond_n,

                p_c,

                p_a

            )



            result = scalers[prop]["y"].inverse_transform(

                pred.cpu()
                .numpy()
                .reshape(-1,1)

            )[0][0]




        return jsonify({

            "status":
            "success",

            "value":
            round(
                float(result),
                4
            )

        })



    except Exception as e:


        print(
            "Prediction Error:",
            e
        )


        return jsonify({

            "status":
            "error",

            "message":
            str(e)

        }),500





if __name__ == "__main__":


    print(
        ">>> IL-GNN预测服务已启动，监听端口5000..."
    )


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )