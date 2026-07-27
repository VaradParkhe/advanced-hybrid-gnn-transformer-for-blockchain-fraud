#!/usr/bin/env python3
"""
plot_roc_model2.py
-------------------
Generates a multiclass ROC curve for Model2 (HybridGNN).
Saves: model2_out/roc_curve.png

Requirements:
 - Model2.py (defines HybridGNN)
 - model2_out/best_model.pt
 - model2_out/normalizer.json
 - dataset CSVs (same ones used in training)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import torch
import torch.nn.functional as F

# Import model class
from Model2 import HybridGNN, create_features

# -------------------------
# Paths
# -------------------------
DATASET_DIR = "./dataset"
ADDR_CSV = "./dataset/synthetic_ethereum_multiclass_dataset_v2.csv"
EDGES_CSV = "./dataset/synthetic_ethereum_edges_v2.csv"
MODEL_DIR = "./model2_out"
MODEL_PATH = "./model2_out/best_model.pt"
ROC_PATH = "./model2_out/roc_curve.png"
METRICS_PATH = "./model2_out/test_metrics.json"

device = torch.device("cpu")


# -------------------------
# Load label map
# -------------------------
if not os.path.exists(METRICS_PATH):
    raise FileNotFoundError("Missing test_metrics.json")

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

label_map = metrics["label_map"]
labels = list(label_map.keys())
num_classes = len(label_map)


# -------------------------
# Load dataset features using same function as training
# -------------------------
class Args:
    addr_csv = ADDR_CSV
    edges_csv = EDGES_CSV
    save_dir = MODEL_DIR
    time_bins = 32

args = Args()
data = create_features(args)

x_static = data["x_static"]
x_time = data["x_time"]
edge_index = data["edge_index"]
y_true = data["y"].cpu().numpy()


# -------------------------
# Load trained model
# -------------------------
model = HybridGNN(
    in_static=x_static.shape[1],
    in_temporal=x_time.shape[2],
    hidden_gnn=64,
    hidden_trans=64,
    gnn_heads=2,
    trans_heads=4,
    trans_layers=2,
    dropout=0.3,
    num_classes=num_classes,
    residual=True
).to(device)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()


# -------------------------
# Get predicted probabilities
# -------------------------
with torch.no_grad():
    logits = model(x_static, x_time, edge_index)
    y_prob = F.softmax(logits, dim=1).cpu().numpy()


# -------------------------
# Plot ROC curves
# -------------------------
def plot_multiclass_roc(y_true, y_prob, labels, save_path):
    y_bin = label_binarize(y_true, classes=list(range(len(labels))))
    n_classes = y_bin.shape[1]

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot
    plt.figure(figsize=(10, 8))

    # Micro and Macro
    plt.plot(fpr["micro"], tpr["micro"],
             label=f"micro-average (AUC={roc_auc['micro']:.3f})", linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label=f"macro-average (AUC={roc_auc['macro']:.3f})", linewidth=2)

    # Each class
    for i, lbl in enumerate(labels):
        plt.plot(fpr[i], tpr[i], lw=2, label=f"{lbl} (AUC={roc_auc[i]:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Model2 Multiclass ROC Curve")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


# Generate curve
plot_multiclass_roc(y_true, y_prob, labels, ROC_PATH)

print(f"ROC curve saved to {ROC_PATH}")
