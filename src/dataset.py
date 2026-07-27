#!/usr/bin/env python3
"""
dataset.py (fixed)
Improved synthetic multi-class Ethereum fraud dataset generator (robust merges)
Author: Your Name (patched)
Date: 2025-11-17
"""
import os
import random
import numpy as np
import pandas as pd
from scipy.stats import entropy

# Config
RNG_SEED = 42
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

N_SAMPLES = 10_000
N_EDGES = 200_000      # larger to create realistic degree distributions
OUTPUT_DIR = r"D:\Model2\dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ['Ponzi', 'Phishing', 'Rug Pull', 'Malicious Contract', 'Legitimate']
LABEL_PROB = [0.08, 0.12, 0.08, 0.12, 0.60]  # tune class ratios

# helper: random address generator
def rand_addr():
    return f"0x{random.getrandbits(160):040x}".lower()

addresses = [rand_addr() for _ in range(N_SAMPLES)]
labels = np.random.choice(LABELS, size=N_SAMPLES, p=LABEL_PROB)
df = pd.DataFrame({'address': addresses, 'label': labels})

# Base network/graph-like features (more realistic distributions)
df['degree_centrality'] = np.random.beta(1.2, 5.0, N_SAMPLES)
df['clustering_coefficient'] = np.random.beta(1.0, 4.0, N_SAMPLES)
df['pagerank'] = np.random.exponential(scale=0.01, size=N_SAMPLES)
df['betweenness'] = np.random.exponential(scale=0.01, size=N_SAMPLES)

# Behavioral / temporal
df['burstiness'] = np.random.exponential(scale=0.5, size=N_SAMPLES)
df['avg_tx_interval'] = np.random.exponential(scale=3600, size=N_SAMPLES)  # seconds
df['lifetime_days'] = np.random.exponential(scale=180, size=N_SAMPLES)
df['tx_rate'] = np.random.exponential(scale=5, size=N_SAMPLES)

# code-like features (simulate distribution)
df['opcode_entropy'] = np.random.normal(loc=3.0, scale=1.5, size=N_SAMPLES).clip(0, 8)
df['code_similarity'] = np.random.rand(N_SAMPLES)
df['verified_contract'] = np.random.choice([0, 1], size=N_SAMPLES, p=[0.85, 0.15])

df['avg_in_value'] = np.random.exponential(scale=0.5, size=N_SAMPLES)
df['avg_out_value'] = np.random.exponential(scale=0.5, size=N_SAMPLES)
df['total_tx_count'] = np.random.randint(1, 20000, N_SAMPLES)

# timestamps
t0 = 1500000000
t1 = 1700000000
df['created_at'] = np.random.randint(t0, t0 + 5_000_000, size=N_SAMPLES)
df['last_active'] = df['created_at'] + np.random.randint(1000, 20_000_000, size=N_SAMPLES)

# class-specific behavior (more realistic patterns)
def amplify(label, feature, factor):
    df.loc[df['label'] == label, feature] *= factor

# Ponzi: sustained outflows, long life, many victims
amplify('Ponzi', 'tx_rate', 2.0)
amplify('Ponzi', 'avg_out_value', 3.0)
df.loc[df['label']=='Ponzi', 'lifetime_days'] *= 2.0
df.loc[df['label']=='Ponzi', 'degree_centrality'] *= 2.0

# Phishing: many small inbound tx, short life after attack
amplify('Phishing', 'burstiness', 2.5)
amplify('Phishing', 'avg_in_value', 3.0)
df.loc[df['label']=='Phishing', 'lifetime_days'] *= 0.5
df.loc[df['label']=='Phishing', 'tx_rate'] *= 0.7

# Rug Pull: sudden high inbound, then stop (short life)
amplify('Rug Pull', 'burstiness', 3.0)
amplify('Rug Pull', 'avg_in_value', 2.5)
df.loc[df['label']=='Rug Pull', 'lifetime_days'] *= 0.2
df.loc[df['label']=='Rug Pull', 'tx_rate'] *= 1.5

# Malicious contracts: high opcode entropy, low similarity, unverified
amplify('Malicious Contract', 'opcode_entropy', 1.8)
df.loc[df['label']=='Malicious Contract', 'code_similarity'] *= 0.2
df.loc[df['label']=='Malicious Contract', 'verified_contract'] = 0
df.loc[df['label']=='Malicious Contract', 'pagerank'] *= 1.8

# Legitimate: lower burstiness, lower variance
df.loc[df['label']=='Legitimate', 'burstiness'] *= 0.6
df.loc[df['label']=='Legitimate', 'tx_rate'] *= 0.8
df.loc[df['label']=='Legitimate', 'opcode_entropy'] *= 0.8

# fraud_source
fraud_sources = ['Etherscan', 'CryptoScamDB', 'PhishFort', 'ETL']
df['fraud_source'] = 'N/A'
mask_fraud = df['label'] != 'Legitimate'
df.loc[mask_fraud, 'fraud_source'] = np.random.choice(fraud_sources, size=mask_fraud.sum())

# Edge generation with preferential attachment (makes hubs like exchanges)
print("Generating edges (preferential attachment style)...")
edge_records = []
degrees = np.ones(N_SAMPLES, dtype=np.float64)  # initial attractiveness

addr_list = df['address'].tolist()
addr_index = {a:i for i,a in enumerate(addr_list)}

# precompute probabilities arrays (we will recompute as needed)
tx_rate_arr = df['tx_rate'].values
pagerank_arr = df['pagerank'].values

for i in range(N_EDGES):
    # choose src biased by tx_rate and degree
    p_src = (tx_rate_arr + degrees)
    p_src = p_src / p_src.sum()
    idx_src = np.random.choice(N_SAMPLES, p=p_src)

    # choose dst biased by pagerank and degree
    p_dst = (pagerank_arr + degrees)
    p_dst = p_dst / p_dst.sum()
    idx_dst = np.random.choice(N_SAMPLES, p=p_dst)

    if idx_src == idx_dst:
        idx_dst = (idx_dst + 1) % N_SAMPLES

    src = addr_list[idx_src]
    dst = addr_list[idx_dst]

    # temporal & value pattern depends on src label
    label = df.iloc[idx_src]['label']
    if label == 'Ponzi':
        ts = int(np.random.normal(t1 - 50_000, 60_000))
        value = abs(np.random.exponential(1.0) * 2.0)
    elif label == 'Phishing':
        ts = int(np.random.normal(t1 - 200_000, 30_000))
        value = abs(np.random.exponential(0.5) * 0.5)
    elif label == 'Rug Pull':
        ts = int(np.random.uniform(t1 - 10_000, t1))
        value = abs(np.random.exponential(3.0))
    elif label == 'Malicious Contract':
        ts = int(np.random.randint(t0, t1))
        value = abs(np.random.uniform(0.01, 5.0))
    else:
        ts = int(np.random.randint(t0, t1))
        value = abs(np.random.exponential(1.0))

    edge_records.append((src, dst, float(value), int(ts)))

    # update degrees to simulate preferential growth
    degrees[idx_src] += 1.0
    degrees[idx_dst] += 0.5

edges = pd.DataFrame(edge_records, columns=['src', 'dst', 'value', 'timeStamp'])

# Make sure all addresses are lowercase
edges['src'] = edges['src'].astype(str).str.lower()
edges['dst'] = edges['dst'].astype(str).str.lower()
df['address'] = df['address'].astype(str).str.lower()

# Graph-derived features
print("Computing graph features...")

# Properly create feature DataFrames with a column named 'address' before merging.
in_deg = edges['dst'].value_counts().rename_axis('address').reset_index(name='in_degree')
out_deg = edges['src'].value_counts().rename_axis('address').reset_index(name='out_degree')

unique_dst = edges.groupby('src')['dst'].nunique().reset_index().rename(columns={'src':'address', 'dst':'unique_receivers'})
unique_src = edges.groupby('dst')['src'].nunique().reset_index().rename(columns={'dst':'address', 'src':'unique_senders'})

avg_tx_val = edges.groupby('src')['value'].mean().reset_index().rename(columns={'src':'address', 'value':'avg_tx_value'})
std_tx_val = edges.groupby('src')['value'].std().fillna(0).reset_index().rename(columns={'src':'address', 'value':'std_tx_value'})

feature_frames = [in_deg, out_deg, unique_dst, unique_src, avg_tx_val, std_tx_val]

# Merge them safely; if a frame doesn't have 'address' we rename a likely column to 'address'
for feat in feature_frames:
    if 'address' not in feat.columns:
        # try common fallbacks
        for col in ['src', 'dst', 'index', feat.columns[0] if len(feat.columns)>0 else None]:
            if col in feat.columns:
                feat.rename(columns={col: 'address'}, inplace=True)
                break
    # ensure dtype matches df
    if 'address' in feat.columns:
        feat['address'] = feat['address'].astype(str).str.lower()
    df = df.merge(feat, on='address', how='left')

graph_cols = ['in_degree', 'out_degree', 'unique_receivers', 'unique_senders', 'avg_tx_value', 'std_tx_value']
df[graph_cols] = df[graph_cols].fillna(0.0)

# NEW FEATURES
# tx_entropy per-node: simulate from Dirichlet per node
dirich = np.random.dirichlet(alpha=np.ones(6), size=N_SAMPLES)
tx_entropy = entropy(dirich.T, base=2)
df['tx_entropy'] = tx_entropy
df['victim_ratio'] = df['in_degree'] / (df['out_degree'] + 1.0)
life_span = (df['last_active'] - df['created_at']).astype(float)
max_span = life_span.max() if life_span.max() > 0 else 1.0
df['temporal_phase'] = life_span / max_span

# Robust normalization (median & iqr) for selected numeric columns
num_to_scale = graph_cols + ['tx_entropy', 'victim_ratio', 'temporal_phase',
                             'tx_rate', 'burstiness', 'avg_in_value', 'avg_out_value',
                             'opcode_entropy', 'total_tx_count']
for c in num_to_scale:
    if c in df.columns:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = max(1e-6, q3 - q1)
        df[c] = (df[c] - df[c].median()) / iqr
        df[c] = df[c].clip(-10.0, 10.0)

# Save
print("Saving dataset...")
out_nodes = os.path.join(OUTPUT_DIR, "synthetic_ethereum_multiclass_dataset_v2.csv")
out_edges = os.path.join(OUTPUT_DIR, "synthetic_ethereum_edges_v2.csv")
df.to_csv(out_nodes, index=False)
edges.to_csv(out_edges, index=False)

try:
    import pyarrow
    df.to_parquet(out_nodes.replace(".csv", ".parquet"), index=False)
    edges.to_parquet(out_edges.replace(".csv", ".parquet"), index=False)
except Exception:
    pass

print("Done. Files saved to:", OUTPUT_DIR)
print("\nLabel counts:\n", df['label'].value_counts())
