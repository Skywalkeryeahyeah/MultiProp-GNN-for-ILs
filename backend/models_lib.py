import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Dropout, BatchNorm1d
from torch_geometric.nn import GATConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors

# --- 核心特征提取器 ---
class FinalFeaturizer:
    def __init__(self, fp_size=512):
        self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
        self.en_map = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}

    def get_features(self, smiles):
        if not smiles or str(smiles).strip() == "": return None
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None: return None
        node_feats = []
        for a in mol.GetAtoms():
            hyb = a.GetHybridization()
            hyb_val = 1 if hyb == Chem.HybridizationType.SP else 2 if hyb == Chem.HybridizationType.SP2 else 3 if hyb == Chem.HybridizationType.SP3 else 0
            node_feats.append([
                a.GetAtomicNum(), a.GetFormalCharge(), 1 if a.IsInRing() else 0,
                a.GetTotalNumHs(), a.GetMass() * 0.01, self.en_map.get(a.GetAtomicNum(), 2.0), hyb_val
            ])
        x = torch.tensor(node_feats, dtype=torch.float)
        edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
        if not edges:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            edges += [[j, i] for i, j in edges]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        fp = torch.tensor(list(self.fp_gen.GetFingerprint(mol)), dtype=torch.float)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        return x, edge_index, fp, mw, tpsa

# --- 密度与热容模型 (对应 519 维输入) ---
class ILCpModel(torch.nn.Module):
    def __init__(self, node_in_dim=7):
        super(ILCpModel, self).__init__()
        self.cat_conv = GATConv(node_in_dim, 48, heads=8)
        self.ani_conv = GATConv(node_in_dim, 48, heads=8)
        self.post_conv = Linear(48 * 8, 96)
        self.fusion = Sequential(Linear(1219, 512), ReLU(), BatchNorm1d(512), Dropout(0.4))
        self.final_regressor = Sequential(Linear(519, 128), ReLU(), Linear(128, 1))

    def forward(self, c_b, a_b, c_fp, a_fp, cond, c_p, a_p):
        c_g = F.elu(self.post_conv(global_mean_pool(self.cat_conv(c_b.x, c_b.edge_index), c_b.batch)))
        a_g = F.elu(self.post_conv(global_mean_pool(self.ani_conv(a_b.x, a_b.edge_index), a_b.batch)))
        combined = torch.cat([c_g, a_g, c_fp, a_fp, cond], dim=1)
        mid = self.fusion(combined)
        final_in = torch.cat([mid, cond, c_p, a_p], dim=1)
        return self.final_regressor(final_in)

# --- 溶解度模型 (对应 522 维输入) ---
class ILSolubilityModel(torch.nn.Module):
    def __init__(self, node_in_dim=7):
        super(ILSolubilityModel, self).__init__()
        self.cat_conv = GATConv(node_in_dim, 48, heads=8)
        self.ani_conv = GATConv(node_in_dim, 48, heads=8)
        self.post_conv = Linear(48 * 8, 96)
        self.fusion = Sequential(Linear(1219, 512), ReLU(), BatchNorm1d(512), Dropout(0.4))
        self.final_regressor = Sequential(Linear(522, 128), ReLU(), Linear(128, 1))

    def forward(self, c_b, a_b, c_fp, a_fp, cond, c_p, a_p):
        c_g = F.elu(self.post_conv(global_mean_pool(self.cat_conv(c_b.x, c_b.edge_index), c_b.batch)))
        a_g = F.elu(self.post_conv(global_mean_pool(self.ani_conv(a_b.x, a_b.edge_index), a_b.batch)))
        combined = torch.cat([c_g, a_g, c_fp, a_fp, cond], dim=1)
        mid = self.fusion(combined)
        final_in = torch.cat([mid, cond, cond, c_p, a_p], dim=1)
        return self.final_regressor(final_in)

# --- 表面张力模型 (对应 520 维输入) ---
class ILSurfaceTensionModel(torch.nn.Module):
    def __init__(self, node_in_dim=7):
        super(ILSurfaceTensionModel, self).__init__()
        self.cat_conv = GATConv(node_in_dim, 48, heads=8)
        self.ani_conv = GATConv(node_in_dim, 48, heads=8)
        self.post_conv = Linear(48 * 8, 96)
        self.fusion = Sequential(Linear(1218, 512), ReLU(), BatchNorm1d(512), Dropout(0.4))
        self.final_regressor = Sequential(Linear(520, 128), ReLU(), Linear(128, 1))

    def forward(self, c_b, a_b, c_fp, a_fp, cond, c_p, a_p):
        c_g = F.elu(self.post_conv(global_mean_pool(self.cat_conv(c_b.x, c_b.edge_index), c_b.batch)))
        a_g = F.elu(self.post_conv(global_mean_pool(self.ani_conv(a_b.x, a_b.edge_index), a_b.batch)))
        combined = torch.cat([c_g, a_g, c_fp, a_fp, cond], dim=1)
        mid = self.fusion(combined)
        final_in = torch.cat([mid, cond, cond, c_p, a_p], dim=1)
        return self.final_regressor(final_in)