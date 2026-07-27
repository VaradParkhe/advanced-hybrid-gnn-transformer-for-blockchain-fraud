#!/usr/bin/env python3
"""
inference_realtime.py
Real-time single-node inference for Model2 (HybridGNN + Temporal Transformer).

Requirements:
 - Model2.py (defines class HybridGNN with signature:
       __init__(in_static, in_temporal, hidden_gnn, hidden_trans,
                gnn_heads, trans_heads, trans_layers, dropout, num_classes, residual=True)
 - model2_out/best_model.pt
 - model2_out/normalizer.json
 - model2_out/test_metrics.json

This script:
 - loads normalizer & label_map
 - instantiates HybridGNN with matching defaults
 - keeps an in-memory node_stats store
 - exposes predict_edge(src,dst,value,timestamp) which updates state and returns predictions
"""

import os
import sys
import time
import json
from collections import defaultdict
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

# Ensure current dir is on sys.path so "from Model2 import HybridGNN" works
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the model class from your training script
try:
    from Model2 import HybridGNN
except Exception as e:
    raise ImportError("Could not import HybridGNN from Model2.py. "
                      "Make sure Model2.py is in the same folder and defines class HybridGNN.") from e

# ------------------------
# Paths (adjust if needed)
# ------------------------
MODEL_DIR = os.path.join(PROJECT_ROOT, "model2_out")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
NORMALIZER_PATH = os.path.join(MODEL_DIR, "normalizer.json")
METRICS_PATH = os.path.join(MODEL_DIR, "test_metrics.json")

# ------------------------
# Device
# ------------------------
device = torch.device("cpu")

# ------------------------
# Load normalizer (supports two possible key styles)
# ------------------------
if not os.path.exists(NORMALIZER_PATH):
    raise FileNotFoundError(f"normalizer.json not found at {NORMALIZER_PATH}")

with open(NORMALIZER_PATH, "r") as f:
    norm = json.load(f)

# Support both {"cols": [...], "mean": [...], "std": [...]} and {"columns": [...], "mean": [...], "std": [...]}
if "cols" in norm:
    static_cols = norm["cols"]
elif "columns" in norm:
    static_cols = norm["columns"]
elif "columns" in norm or "cols" in norm:
    static_cols = norm.get("cols") or norm.get("columns")
else:
    # fallback: maybe normalizer saved as {"columns":..., "mean":..., "std":...}
    raise KeyError("normalizer.json missing 'cols' or 'columns' key.")

mean = torch.tensor(norm["mean"], dtype=torch.float32) if "mean" in norm else torch.tensor(norm.get("means", []), dtype=torch.float32)
std = torch.tensor(norm["std"], dtype=torch.float32) if "std" in norm else torch.tensor(norm.get("stds", []), dtype=torch.float32)

if mean.numel() != len(static_cols) or std.numel() != len(static_cols):
    # defensive: allow broadcasting if scalars, else warn
    if mean.numel() == 1:
        mean = mean.repeat(len(static_cols))
    if std.numel() == 1:
        std = std.repeat(len(static_cols))
    if mean.numel() != len(static_cols) or std.numel() != len(static_cols):
        raise ValueError("normalizer mean/std length mismatch with static_cols length.")

# ------------------------
# Load label_map from metrics
# ------------------------
if not os.path.exists(METRICS_PATH):
    raise FileNotFoundError(f"test_metrics.json not found at {METRICS_PATH}")

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

if "label_map" not in metrics:
    raise KeyError("test_metrics.json missing 'label_map' key (expected mapping label->index).")

label_map = metrics["label_map"]            # e.g. {"Ponzi":0, "Phishing":1, ...}
# Build inverse mapping index->label
inv_label_map = {int(v): k for k, v in label_map.items()}

num_classes = len(label_map)

# ------------------------
# Instantiate model (use sensible defaults, must match training)
# If you used different hyperparameters when training, update these accordingly.
# ------------------------
# Defaults used in training script: hidden_gnn=64, hidden_trans=64, gnn_heads=2, trans_heads=4, trans_layers=2, dropout=0.3
hidden_gnn = 64
hidden_trans = 64
gnn_heads = 2
trans_heads = 4
trans_layers = 2
dropout = 0.3

model = HybridGNN(
    in_static=len(static_cols),
    in_temporal=2,               # count + avg_value channels
    hidden_gnn=hidden_gnn,
    hidden_trans=hidden_trans,
    gnn_heads=gnn_heads,
    trans_heads=trans_heads,
    trans_layers=trans_layers,
    dropout=dropout,
    num_classes=num_classes,
    residual=True
).to(device)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"best_model.pt not found at {MODEL_PATH}")

# load state
state = torch.load(MODEL_PATH, map_location=device)
# If the checkpoint was saved as state_dict directly it's fine; if saved with wrapper keys (e.g. {"model":...}) handle gracefully
if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
    # probably state_dict with module. prefixes
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
    state = new_state

# If file contains multiple keys (e.g. {"model_state":...}), try to find best candidate
if not any(isinstance(v, torch.Tensor) for v in state.values()):
    # try common wrappers
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state and isinstance(state["model"], dict):
        state = state["model"]

model.load_state_dict(state)
model.eval()

print("🚀 inference_realtime.py — model loaded and ready")

# ------------------------
# In-memory node store
# ------------------------
node_stats = defaultdict(lambda: {
    "in_degree": 0,
    "out_degree": 0,
    "unique_senders": set(),
    "unique_receivers": set(),
    "avg_in_value": [],
    "avg_out_value": [],
    "tx_rate": [],
    "timestamps": []
})

# ------------------------
# Helper: build per-address features and return tensors
# ------------------------
def build_features(address: str, bins: int = 32, window_seconds: int = 6 * 3600) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (x_static_tensor (1,F), x_time_tensor (1,T,2), edge_index (2, E))
    For single-node inference we use a self-loop edge_index [[0],[0]] so GAT can operate on a single node.
    """
    ns = node_stats[address]

    feat = {
        "in_degree": float(ns["in_degree"]),
        "out_degree": float(ns["out_degree"]),
        "tx_rate": float(np.mean(ns["tx_rate"])) if ns["tx_rate"] else 0.0,
        "burstiness": float(np.std(ns["timestamps"])) if len(ns["timestamps"]) > 2 else 0.0,
        "avg_in_value": float(np.mean(ns["avg_in_value"])) if ns["avg_in_value"] else 0.0,
        "avg_out_value": float(np.mean(ns["avg_out_value"])) if ns["avg_out_value"] else 0.0,
        "opcode_entropy": 0.0,
        "code_similarity": 0.0,
        "tx_entropy": 0.0,
        "temporal_phase": 0.0,
        "victim_ratio": float(ns["in_degree"]) / (float(ns["out_degree"]) + 1.0),
        "unique_receivers": float(len(ns["unique_receivers"])),
        "unique_senders": float(len(ns["unique_senders"])),
        "total_tx_count": float(ns["in_degree"] + ns["out_degree"])
    }

    # Create ordered static vector according to static_cols
    vec = [feat.get(c, 0.0) for c in static_cols]
    x = torch.tensor(vec, dtype=torch.float32)
    # normalize (use saved mean/std)
    x = (x - mean) / (std + 1e-6)
    x = x.unsqueeze(0).to(device)  # shape (1, F)

    # Temporal bins
    now = time.time()
    X_time = np.zeros((bins, 2), dtype=np.float32)
    timestamps = ns["timestamps"]
    values = ns["avg_out_value"]
    if timestamps and len(timestamps) == len(values):
        for t, v in zip(timestamps, values):
            dt = now - float(t)
            bin_id = int((dt / float(window_seconds)) * bins)
            if bin_id < 0:
                bin_id = 0
            if bin_id >= bins:
                bin_id = bins - 1
            X_time[bin_id, 0] += 1.0
            X_time[bin_id, 1] += float(v)

    xt = torch.tensor(X_time, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, 2)

    # self-loop edge index so GAT can run on single node
    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)

    return x, xt, edge_index


# ------------------------
# Update state on incoming tx
# ------------------------
def update_state(src: str, dst: str, value: float, timestamp: float = None) -> None:
    if timestamp is None:
        timestamp = time.time()
    # ensure lowercase keys to match dataset addresses convention
    src = src.lower()
    dst = dst.lower()

    ns = node_stats[src]
    ns["out_degree"] += 1
    ns["unique_receivers"].add(dst)
    ns["avg_out_value"].append(float(value))
    ns["timestamps"].append(float(timestamp))
    ns["tx_rate"].append(1.0)

    nd = node_stats[dst]
    nd["in_degree"] += 1
    nd["unique_senders"].add(src)
    nd["avg_in_value"].append(float(value))
    nd["timestamps"].append(float(timestamp))
    nd["tx_rate"].append(1.0)


# ------------------------
# Predict one address
# ------------------------
def predict_address(address: str) -> Dict[str, Any]:
    address = address.lower()
    x, xt, ei = build_features(address)
    with torch.no_grad():
        logits = model(x, xt, ei)             # (1, num_classes)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_label = inv_label_map.get(pred_idx, "UNKNOWN")
    # build probability dict label->prob
    prob_dict = {lbl: float(probs[idx]) for lbl, idx in label_map.items()}
    return {"address": address, "prediction": pred_label, "probabilities": prob_dict}


# ------------------------
# Predict from incoming tx (updates state, returns both src & dst predictions)
# ------------------------
def predict_edge(src: str, dst: str, value: float, timestamp: float = None) -> Dict[str, Any]:
    update_state(src, dst, value, timestamp)
    return {"src_prediction": predict_address(src), "dst_prediction": predict_address(dst)}


# ------------------------
# CLI / quick demo
# ------------------------
if __name__ == "__main__":
    print("Demo: simple synthetic transaction -> predictions")
    demo_src = "0x1111111111111111111111111111111111111111"
    demo_dst = "0x2222222222222222222222222222222222222222"
    ts = time.time()
    # simulate a few transactions to populate temporal bins
    for i in range(6):
        predict_edge(demo_src, demo_dst, value=0.1 + i * 0.05, timestamp=ts - i * 60)
    out = predict_edge(demo_src, demo_dst, value=0.5, timestamp=ts)
    print(json.dumps(out, indent=2))
