"""
Diagnostic Visualization for Synthetic Ethereum Dataset (V2)
------------------------------------------------------------
Generates plots showing class separability and feature distributions.
Compatible with your updated synthetic_ethereum_multiclass_dataset_v2.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

DATA_PATH = r"D:\Model2\dataset\synthetic_ethereum_multiclass_dataset_v2.csv"
SAVE_DIR = r"D:\Model2\dataset\plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load dataset
data = pd.read_csv(DATA_PATH)

# Sampling (avoid overload)
sample = data.sample(min(5000, len(data)), random_state=42)

# Color palette
palette = {
    'Ponzi': '#D62828',
    'Phishing': '#F77F00',
    'Rug Pull': '#E9C46A',
    'Malicious Contract': '#457B9D',
    'Legitimate': '#2A9D8F'
}

plt.rcParams["figure.dpi"] = 150
plt.style.use("seaborn-v0_8-whitegrid")


# ================================================================
# 1️⃣ Transaction Rate vs Avg Tx Value
# ================================================================
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=sample,
    x='tx_rate', y='avg_tx_value', hue='label',
    palette=palette, alpha=0.6, s=35
)
plt.title("Transaction Rate vs Average Transaction Value", fontsize=14, weight='bold')
plt.xlabel("Transaction Rate (normalized)")
plt.ylabel("Average Tx Value (normalized)")
plt.legend(title="Class", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "tx_rate_vs_value.png"))
plt.close()


# ================================================================
# 2️⃣ Unique Receivers vs Unique Senders
# ================================================================
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=sample,
    x='unique_receivers', y='unique_senders',
    hue='label', palette=palette, alpha=0.6, s=35
)
plt.title("Counterparty Diversity by Class", fontsize=14, weight='bold')
plt.xlabel("Unique Receivers (normalized)")
plt.ylabel("Unique Senders (normalized)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "counterparty_diversity.png"))
plt.close()


# ================================================================
# 3️⃣ Victim Ratio Distribution
# ================================================================
plt.figure(figsize=(8, 5))
sns.kdeplot(
    data=sample,
    x='victim_ratio',
    hue='label',
    palette=palette,
    common_norm=False,
    alpha=0.5,
    linewidth=2
)
plt.title("Victim Ratio Distribution", fontsize=14, weight='bold')
plt.xlabel("Victim Ratio (in_degree / out_degree)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "victim_ratio_distribution.png"))
plt.close()


# ================================================================
# 4️⃣ Opcode Entropy Distribution
# ================================================================
plt.figure(figsize=(8, 5))
sns.kdeplot(
    data=sample,
    x='opcode_entropy',
    hue='label',
    palette=palette,
    common_norm=False,
    alpha=0.5
)
plt.title("Opcode Entropy Distribution Across Classes")
plt.xlabel("Opcode Entropy")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "opcode_entropy.png"))
plt.close()


# ================================================================
# 5️⃣ Temporal Phase Distribution
# ================================================================
plt.figure(figsize=(8, 5))
sns.kdeplot(
    data=sample,
    x='temporal_phase',
    hue='label',
    palette=palette,
    common_norm=False
)
plt.title("Temporal Phase Distribution (Lifecycle Position)")
plt.xlabel("Temporal Phase (0 → newly created, 1 → old account)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "temporal_phase.png"))
plt.close()


# ================================================================
# 6️⃣ 2D Density: Pagerank vs Degree Centrality
# ================================================================
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=sample,
    x='pagerank',
    y='degree_centrality',
    hue='label',
    palette=palette,
    alpha=0.6,
    s=35
)
plt.title("Pagerank vs Degree Centrality", fontsize=14, weight='bold')
plt.xlabel("Pagerank (normalized)")
plt.ylabel("Degree Centrality (normalized)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "pagerank_vs_degree.png"))
plt.close()


# ================================================================
# 7️⃣ Boxplot: Burstiness per Class
# ================================================================
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=sample,
    x='label',
    y='burstiness',
    palette=palette
)
plt.title("Burstiness Distribution per Fraud Type")
plt.xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "burstiness_boxplot.png"))
plt.close()


# ================================================================
# 8️⃣ Correlation Heatmap (All Features)
# ================================================================
plt.figure(figsize=(12, 9))
corr = data.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "correlation_heatmap.png"))
plt.close()

print(f"✅ Diagnostic plots saved to: {SAVE_DIR}")
