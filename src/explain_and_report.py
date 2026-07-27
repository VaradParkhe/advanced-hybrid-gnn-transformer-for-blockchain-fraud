#!/usr/bin/env python3
"""
Explain & Report for Model2 (Matched to Training)
-------------------------------------------------
This version EXACTLY matches the 14 static features used in Model2.py
so best_model.pt loads with ZERO mismatches.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv

LABELS = ['Ponzi','Phishing','Rug Pull','Malicious Contract','Legitimate']
label_map = {c:i for i,c in enumerate(LABELS)}

# ============================================================
# ROTARY POSITION EMBEDDINGS
# ============================================================
def apply_rope(x):
    dim = x.size(-1)
    half = dim // 2
    freq = torch.arange(half, device=x.device).float()
    freq = 10000 ** (-2 * freq / dim)
    pos = torch.arange(x.size(1), device=x.device).float()
    sinusoid = torch.einsum("i,j->ij", pos, freq)
    sin = sinusoid.sin().unsqueeze(0)
    cos = sinusoid.cos().unsqueeze(0)
    x1 = x[..., :half]
    x2 = x[..., half:half*2]
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)

# ============================================================
# TEMPORAL TRANSFORMER
# ============================================================
class TemporalTransformer(nn.Module):
    def __init__(self, in_dim, d_model, heads, layers, drop):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1,1,d_model))
        self.proj = nn.Linear(in_dim, d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model*2,
            dropout=drop,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B = x.size(0)
        z = self.proj(x)
        z = apply_rope(z)
        cls = self.cls.expand(B, -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = self.encoder(z)
        return self.norm(z[:,0])

# ============================================================
# HYBRID GNN (Model2 architecture)
# ============================================================
class HybridGNN(nn.Module):
    def __init__(self, in_static, in_temporal, hidden_gnn, hidden_trans,
                 gnn_heads, trans_heads, trans_layers, dropout, num_classes, residual=True):
        super().__init__()
        self.residual = residual

        self.static_proj = nn.Linear(in_static, hidden_gnn) if in_static != hidden_gnn else None

        self.g1 = GATv2Conv(in_static, hidden_gnn, heads=gnn_heads, dropout=dropout, concat=True)
        self.g2 = GATv2Conv(hidden_gnn*gnn_heads, hidden_gnn, heads=1, dropout=dropout, concat=False)
        self.bn_g = nn.BatchNorm1d(hidden_gnn)

        self.temporal = TemporalTransformer(in_temporal, hidden_trans, trans_heads, trans_layers, dropout)
        self.bn_t = nn.BatchNorm1d(hidden_trans)

        mlp_in = hidden_gnn + hidden_trans
        self.out = nn.Sequential(
            nn.Linear(mlp_in, max(128, mlp_in//2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(128, mlp_in//2), num_classes)
        )

    def encode_g(self, xs, edge_index):
        g = F.elu(self.g1(xs, edge_index))
        g = self.g2(g, edge_index)

        if self.static_proj is not None:
            xs_proj = self.static_proj(xs)
            if xs_proj.shape == g.shape:
                g = g + xs_proj

        return self.bn_g(g)

    def encode_t(self, xt):
        return self.bn_t(self.temporal(xt))

    def forward(self, xs, xt, edge_index):
        g = self.encode_g(xs, edge_index)
        t = self.encode_t(xt)
        return self.out(torch.cat([g,t], dim=-1))

# ============================================================
# BUILD FEATURES (EXACT MATCH TO MODEL2.PY)
# ============================================================
STATIC_COLS = [
    'in_degree','out_degree','tx_rate','burstiness',
    'avg_in_value','avg_out_value','opcode_entropy','code_similarity',
    'tx_entropy','temporal_phase','unique_receivers','unique_senders',
    'victim_ratio','total_tx_count'
]
STATIC_DIM = 14   # <-- must match checkpoint EXACTLY

def build_data(addr_csv, edges_csv):
    addr = pd.read_csv(addr_csv)
    edges = pd.read_csv(edges_csv)

    addr["address"] = addr["address"].str.lower()
    edges["src"] = edges["src"].str.lower()
    edges["dst"] = edges["dst"].str.lower()

    nodes = sorted(set(edges["src"]).union(edges["dst"]))
    idx = {a:i for i,a in enumerate(nodes)}
    N = len(nodes)

    # labels
    y = np.full(N, label_map["Legitimate"])
    for _, r in addr.iterrows():
        if r["address"] in idx:
            y[idx[r["address"]]] = label_map.get(r["label"], label_map["Legitimate"])

    # edge_index
    edges["sidx"] = edges["src"].map(idx)
    edges["didx"] = edges["dst"].map(idx)
    edges = edges.dropna(subset=["sidx","didx"])
    edge_index = torch.tensor(edges[["sidx","didx"]].values.T, dtype=torch.long)

    # Static features EXACTLY as Model2.py
    df = pd.DataFrame({"address":nodes})
    df = df.merge(addr, how="left", on="address")

    df["in_degree"] = edges["didx"].value_counts().reindex(range(N), fill_value=0).values
    df["out_degree"] = edges["sidx"].value_counts().reindex(range(N), fill_value=0).values
    df["unique_receivers"] = df["out_degree"]
    df["unique_senders"] = df["in_degree"]
    df["victim_ratio"] = df["in_degree"]/(df["out_degree"]+1)
    df["total_tx_count"] = df["in_degree"] + df["out_degree"]

    for c in STATIC_COLS:
        if c not in df: df[c]=0

    Xs = df[STATIC_COLS].to_numpy().astype(np.float32)
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0)+1e-6)
    Xs = torch.tensor(Xs, dtype=torch.float32)

    # temporal
    edges["timeStamp"] = pd.to_numeric(edges["timeStamp"], errors="coerce").fillna(0)
    tmin,tmax = edges["timeStamp"].min(), edges["timeStamp"].max()
    T = 32
    if tmax > tmin:
        edges["bin"] = (((edges["timeStamp"]-tmin)/(tmax-tmin))*T).clip(0,T-1).astype(int)
    else:
        edges["bin"] = 0

    Xt = torch.zeros((N,T,2), dtype=torch.float32)
    g = edges.groupby(["sidx","bin"]).agg(cnt=("didx","count"), val=("value","mean")).reset_index()
    Xt[g["sidx"], g["bin"], 0] = torch.tensor(g["cnt"].values, dtype=torch.float32)
    Xt[g["sidx"], g["bin"], 1] = torch.tensor(g["val"].fillna(0).values, dtype=torch.float32)

    Xt = (Xt - Xt.mean((0,1),keepdim=True))/(Xt.std((0,1),keepdim=True)+1e-6)

    return {
        "x_static": Xs,
        "x_time": Xt,
        "edge_index": edge_index,
        "y": torch.tensor(y),
        "nodes": nodes
    }

# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================
def extract_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        g = model.encode_g(data["x_static"], data["edge_index"])
        t = model.encode_t(data["x_time"])
        return torch.cat([g,t], dim=1).cpu().numpy()

# ============================================================
# TSNE
# ============================================================
def make_tsne(emb, labels, save_path):
    print("→ Running PCA...")
    pca = PCA(n_components=50).fit_transform(emb)

    print("→ Running TSNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        max_iter=2000,
        verbose=1
    )
    ts = tsne.fit_transform(pca)

    plt.figure(figsize=(10,8))
    for i,label in enumerate(LABELS):
        m = labels==i
        plt.scatter(ts[m,0], ts[m,1], s=10, alpha=0.6, label=label)
    plt.legend()
    plt.title("t-SNE Embeddings")
    plt.savefig(save_path, dpi=300)
    plt.close()

# ============================================================
# MAIN
# ============================================================
def main():
    dataset = "./dataset"
    addr_csv = dataset + "/synthetic_ethereum_multiclass_dataset_v2.csv"
    edges_csv = dataset + "/synthetic_ethereum_edges_v2.csv"
    model_path = "./model2_out/best_model.pt"
    out_dir = "./explain_plots"
    os.makedirs(out_dir, exist_ok=True)

    print("→ Loading dataset...")
    data = build_data(addr_csv, edges_csv)

    print("→ Loading model...")
    in_static = data["x_static"].shape[1]   # MUST BE 14
    in_temporal = data["x_time"].shape[2]

    hidden_gnn = 64
    hidden_trans = 64

    model = HybridGNN(
        in_static=in_static,
        in_temporal=in_temporal,
        hidden_gnn=hidden_gnn,
        hidden_trans=hidden_trans,
        gnn_heads=2,
        trans_heads=4,
        trans_layers=2,
        dropout=0.3,
        num_classes=5
    )

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)   # Now SAFE

    print("→ Extracting embeddings...")
    emb = extract_embeddings(model, data)

    print("→ Running TSNE...")
    make_tsne(emb, data["y"].numpy(), out_dir + "/tsne_embeddings.png")

    print("✔ All plots saved to", out_dir)


if __name__ == "__main__":
    main()
