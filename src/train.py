#!/usr/bin/env python3
"""
Model2.py — FIXED STABLE VERSION
--------------------------------
✓ No edge_index errors
✓ No mixup-GNN mismatch
✓ Focal Loss
✓ Rotary Transformer
✓ Residual GNN
✓ Minority oversampling
✓ Normalizer export
✓ Stratified splits
✓ Correct multiclass AUC
"""

import os, json, argparse, logging, random, multiprocessing
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GATv2Conv

# ----------------------------
# CPU SETTINGS
# ----------------------------
cores = os.cpu_count() or multiprocessing.cpu_count()
torch.set_num_threads(max(2, int(cores * 0.8)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ============================================================
# 🏆 FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


# ============================================================
# 🌀 ROTARY POSITION EMBEDDINGS (RoPE)
# ============================================================
def apply_rope(x):
    """
    x: [batch, seq, dim]
    """
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

    x_rot = torch.cat([x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos], dim=-1)

    return x_rot


# ============================================================
# 🔥 TEMPORAL TRANSFORMER (with RoPE)
# ============================================================
class TemporalTransformer(nn.Module):
    def __init__(self, in_dim, d_model, heads, layers, drop):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.proj = nn.Linear(in_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 2,
            dropout=drop,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B = x.size(0)
        z = self.proj(x)
        z = apply_rope(z)
        cls_tok = self.cls.expand(B, -1, -1)
        z = torch.cat([cls_tok, z], dim=1)
        z = self.encoder(z)
        return self.norm(z[:, 0])
# ============================================================
# 🧠 HYBRID GNN (Residual) + helper encoders
# ============================================================
class HybridGNN(nn.Module):
    def __init__(self, in_static, in_temporal, hidden_gnn, hidden_trans,
                 gnn_heads, trans_heads, trans_layers, dropout, num_classes, residual=True):
        super().__init__()
        self.residual = residual
        self.hidden_gnn = hidden_gnn
        self.hidden_trans = hidden_trans

        # optional projection for residual if static dim != hidden_gnn
        self.static_proj = nn.Linear(in_static, hidden_gnn) if in_static != hidden_gnn else None

        # GNN blocks
        self.g1 = GATv2Conv(in_static, hidden_gnn, heads=gnn_heads, dropout=dropout, concat=True)
        self.g2 = GATv2Conv(hidden_gnn * gnn_heads, hidden_gnn, heads=1, dropout=dropout, concat=False)
        self.bn_g = nn.BatchNorm1d(hidden_gnn)

        # Temporal transformer
        self.temporal = TemporalTransformer(in_temporal, hidden_trans, trans_heads, trans_layers, dropout)
        self.bn_t = nn.BatchNorm1d(hidden_trans)

        # Output MLP (combine)
        mlp_in = hidden_gnn + hidden_trans
        self.out = nn.Sequential(
            nn.Linear(mlp_in, max(128, mlp_in // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(128, mlp_in // 2), num_classes)
        )

    def encode_g(self, x_static, edge_index):
        # full-graph GNN encoding (N, hidden_gnn)
        x = F.elu(self.g1(x_static, edge_index))
        x = self.g2(x, edge_index)

        # residual
        if self.residual:
            if self.static_proj is not None:
                res = self.static_proj(x_static)
            else:
                res = x_static
            if res.shape == x.shape:
                x = x + res
        x = self.bn_g(x)
        return x

    def encode_t(self, x_time_batch):
        # x_time_batch: (B, T, C)
        t = self.temporal(x_time_batch)  # (B, hidden_trans)
        t = self.bn_t(t)
        return t

    def forward(self, x_static, x_time, edge_index):
        # default full forward assuming x_time aligns with x_static
        g = self.encode_g(x_static, edge_index)
        t = self.encode_t(x_time)
        h = torch.cat([g, t], dim=-1)
        return self.out(h)


# ============================================================
# 🔀 MixUp for node-level features (temporal + static embeddings)
# ============================================================
def mixup_nodes(xs, xt, y_onehot, alpha=0.4):
    """
    xs: (B, F) - static node embeddings (or projected static)
    xt: (B, T, C) - temporal channels
    y_onehot: (B, K) - soft labels
    """
    if alpha <= 0:
        return xs, xt, y_onehot
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(xs.size(0), device=xs.device)
    xs2 = xs[idx]
    xt2 = xt[idx]
    y2 = y_onehot[idx]
    xs_m = lam * xs + (1 - lam) * xs2
    xt_m = lam * xt + (1 - lam) * xt2
    y_m = lam * y_onehot + (1 - lam) * y2
    return xs_m, xt_m, y_m


# ============================================================
# 🗂 Feature loader: builds X_static, X_time, edge_index, y, label_map
# and saves normalizer.json for inference
# ============================================================
def create_features(args):
    logging.info("Loading CSVs...")
    addr = pd.read_csv(args.addr_csv)
    edges = pd.read_csv(args.edges_csv)

    addr['address'] = addr['address'].astype(str).str.lower()
    edges['src'] = edges['src'].astype(str).str.lower()
    edges['dst'] = edges['dst'].astype(str).str.lower()

    # nodes = all addresses seen in edges
    nodes = sorted(set(edges['src']).union(edges['dst']))
    idx = {a: i for i, a in enumerate(nodes)}
    N = len(nodes)
    logging.info(f"Found {N} nodes in graph.")

    LABELS = ['Ponzi', 'Phishing', 'Rug Pull', 'Malicious Contract', 'Legitimate']
    label_map = {name: i for i, name in enumerate(LABELS)}

    y = torch.full((N,), fill_value=label_map['Legitimate'], dtype=torch.long)

    if 'label' in addr.columns:
        for _, r in addr.iterrows():
            a = r['address']
            if a in idx:
                y[idx[a]] = label_map.get(r['label'], label_map['Legitimate'])

    # map edges to indices and drop missing
    edges['sidx'] = edges['src'].map(idx)
    edges['didx'] = edges['dst'].map(idx)
    edges = edges.dropna(subset=['sidx', 'didx']).copy()
    edges['sidx'] = edges['sidx'].astype(int)
    edges['didx'] = edges['didx'].astype(int)

    edge_index = torch.tensor(edges[['sidx', 'didx']].values.T, dtype=torch.long)

    # build node table (merge addr features if present)
    df = pd.DataFrame({'address': nodes})
    if 'address' in addr.columns:
        df = df.merge(addr, on='address', how='left')

    # graph-derived
    df['in_degree'] = edges['didx'].value_counts().reindex(range(N), fill_value=0).values
    df['out_degree'] = edges['sidx'].value_counts().reindex(range(N), fill_value=0).values
    df['unique_receivers'] = df['out_degree']
    df['unique_senders'] = df['in_degree']
    df['victim_ratio'] = df['in_degree'] / (df['out_degree'] + 1)

    # candidate static columns (ensure existence)
    static_cols = [
        'in_degree', 'out_degree', 'tx_rate', 'burstiness',
        'avg_in_value', 'avg_out_value', 'opcode_entropy', 'code_similarity',
        'tx_entropy', 'temporal_phase', 'unique_receivers', 'unique_senders',
        'victim_ratio', 'total_tx_count'
    ]
    for c in static_cols:
        if c not in df.columns:
            df[c] = 0.0

    # save normalizer for inference
    norm = {"cols": static_cols, "mean": df[static_cols].mean().tolist(), "std": (df[static_cols].std() + 1e-6).tolist()}
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "normalizer.json"), "w") as f:
        json.dump(norm, f, indent=2)

    # static tensor
    Xs = (df[static_cols] - df[static_cols].mean()) / (df[static_cols].std() + 1e-6)
    Xs = torch.tensor(Xs.values, dtype=torch.float32)

    # temporal bins: simple (count, average value)
    edges['timeStamp'] = pd.to_numeric(edges.get('timeStamp', 0), errors='coerce').fillna(0).astype(int)
    tmin, tmax = edges['timeStamp'].min(), edges['timeStamp'].max()
    T = args.time_bins
    if tmax <= tmin:
        edges['bin'] = 0
    else:
        edges['bin'] = ((edges['timeStamp'] - tmin) / (tmax - tmin + 1e-9) * T).astype(int).clip(0, T-1)

    Xt = torch.zeros((N, T, 2), dtype=torch.float32)
    grp = edges.groupby(['sidx', 'bin']).agg(count=('didx', 'count'), avgv=('value', 'mean')).reset_index()
    if not grp.empty:
        Xt[grp['sidx'].values, grp['bin'].values, 0] = torch.tensor(grp['count'].values, dtype=torch.float32)
        Xt[grp['sidx'].values, grp['bin'].values, 1] = torch.tensor(grp['avgv'].fillna(0).values, dtype=torch.float32)

    # normalize temporal channels
    Xt = (Xt - Xt.mean((0, 1), keepdim=True)) / (Xt.std((0, 1), keepdim=True) + 1e-6)

    # return
    return {"x_static": Xs, "x_time": Xt, "edge_index": edge_index, "y": y, "label_map": label_map}
# ============================================================
# -------------------- EVALUATION -----------------------------
# ============================================================
def evaluate(model, data, mask, loss_fn, label_map):
    model.eval()
    with torch.no_grad():
        # Full forward on full graph
        logits_full = model(data["x_static"], data["x_time"], data["edge_index"])
        if mask.sum().item() == 0:
            return float("nan"), float("nan"), "No samples in mask."
        logits = logits_full[mask]
        y_true = data["y"][mask].cpu().numpy()
        loss = float(loss_fn(logits, data["y"][mask]).item())

        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        # multiclass AUC (label_binarize)
        try:
            y_bin = label_binarize(y_true, classes=list(label_map.values()))
            auc = roc_auc_score(y_bin, probs, multi_class="ovr")
        except Exception:
            auc = float("nan")

        report = classification_report(y_true, preds, target_names=list(label_map.keys()), zero_division=0)
        return auc, loss, report


# ============================================================
# -------------------- BATCH / OVERSAMPLING -------------------
# ============================================================
def build_oversampled_batches(train_idx, y_np, batch_size):
    """
    Create oversampled batches where each batch tries to include
    more samples of minority classes by sampling per-class.
    Returns list of numpy arrays of indices.
    """
    from collections import defaultdict
    cls2idx = defaultdict(list)
    for i in train_idx:
        cls2idx[int(y_np[i])].append(int(i))
    classes = list(cls2idx.keys())
    # number of batches = ceil(len(train_idx)/batch_size)
    num_batches = max(1, int(np.ceil(len(train_idx) / batch_size)))
    batches = []
    for _ in range(num_batches):
        sel = []
        # allocate at least one per class (if possible)
        per_class = max(1, batch_size // len(classes))
        for c in classes:
            idxs = cls2idx[c]
            replace = len(idxs) < per_class
            pick = np.random.choice(idxs, size=per_class, replace=replace).tolist()
            sel.extend(pick)
        # fill rest randomly from train_idx
        if len(sel) < batch_size:
            add = np.random.choice(train_idx, size=(batch_size - len(sel)), replace=True).tolist()
            sel.extend(add)
        sel = np.array(sel[:batch_size], dtype=int)
        np.random.shuffle(sel)
        batches.append(sel)
    return batches


# ============================================================
# -------------------- TRAINING LOOP -------------------------
# ============================================================
def main_train(args):
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data = create_features(args)
    y = data["y"]
    label_map = data["label_map"]
    num_classes = len(label_map)

    # stratified splits
    idx_all = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx_all, test_size=0.2, stratify=y.cpu().numpy(), random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=y.cpu().numpy()[train_idx], random_state=args.seed)

    mask_train = torch.zeros_like(y, dtype=torch.bool); mask_train[train_idx] = True
    mask_val = torch.zeros_like(y, dtype=torch.bool); mask_val[val_idx] = True
    mask_test = torch.zeros_like(y, dtype=torch.bool); mask_test[test_idx] = True

    # model
    model = HybridGNN(
        in_static=data["x_static"].shape[1],
        in_temporal=data["x_time"].shape[2],
        hidden_gnn=args.hidden_gnn,
        hidden_trans=args.hidden_trans,
        gnn_heads=args.gnn_heads,
        trans_heads=args.trans_heads,
        trans_layers=args.trans_layers,
        dropout=args.dropout,
        num_classes=num_classes,
        residual=True
    )

    model.to(torch.device("cpu"))

    # base weighting derived from freq (stabilize)
    train_counts = Counter(y[mask_train].cpu().numpy())
    base_weights = torch.ones(num_classes, dtype=torch.float32)
    if train_counts:
        max_cnt = max(train_counts.values())
        for cls, cnt in train_counts.items():
            base_weights[cls] = float(max_cnt / max(1, cnt))

    # optimizer, scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5)

    best_val_auc = -1.0
    best_path = os.path.join(args.save_dir, "best_model.pt")
    os.makedirs(args.save_dir, exist_ok=True)

    y_np = y.cpu().numpy()
    labeled_train_idx = np.where(mask_train.cpu().numpy())[0]

    # training
    for epoch in range(1, args.epochs + 1):
        model.train()

        # dynamic class weights for loss this epoch
        counts_epoch = Counter(y[mask_train].cpu().numpy())
        dyn = torch.ones(num_classes, dtype=torch.float32)
        if counts_epoch:
            inv = {c: 1.0 / counts_epoch.get(c, 1) for c in range(num_classes)}
            max_inv = max(inv.values())
            for c, v in inv.items():
                dyn[c] = float(v / max_inv)
        # blend with base weights
        loss_weights = (1 - args.weight_blend) * base_weights + args.weight_blend * dyn
        loss_weights = loss_weights.to(torch.device("cpu"))
        loss_fn = FocalLoss(alpha=loss_weights, gamma=args.focal_gamma)

        batches = build_oversampled_batches(labeled_train_idx, y_np, args.batch_size)
        epoch_loss = 0.0

        for b in batches:
            b = np.array(b, dtype=int)
            b_idx = torch.tensor(b, dtype=torch.long)

            # 1) compute full-graph GNN embeddings (required so indices align)
            g_full = model.encode_g(data["x_static"], data["edge_index"])  # (N, hidden_gnn)
            g_batch = g_full[b]  # (B, hidden_gnn)

            # 2) temporal batch
            xt_batch = data["x_time"][b]  # (B, T, C)
            labels = data["y"][b]

            # 3) mixup optionally
            if args.mixup_alpha > 0:
                y_onehot = F.one_hot(labels, num_classes).float()
                xs_m, xt_m, y_m = mixup_nodes(g_batch, xt_batch, y_onehot, alpha=args.mixup_alpha)
                # encode temporal mixed
                t_enc = model.encode_t(xt_m)
                h = torch.cat([xs_m, t_enc], dim=-1)
                logits = model.out(h)
                # soft-label loss
                logp = F.log_softmax(logits, dim=1)
                loss = - (y_m * logp).sum(dim=1).mean()
            else:
                # normal
                t_enc = model.encode_t(xt_batch)
                h = torch.cat([g_batch, t_enc], dim=-1)
                logits = model.out(h)
                loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            opt.step()

            epoch_loss += float(loss.item())

        # end epoch evaluation
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_auc, val_loss, _ = evaluate(model, data, mask_val, loss_fn, label_map)
            scheduler.step(val_auc if not np.isnan(val_auc) else 0.0)
            logging.info(f"[E{epoch}] epoch_loss={epoch_loss/len(batches):.4f} ValAUC={val_auc:.4f}")
            if val_auc > best_val_auc and not np.isnan(val_auc):
                best_val_auc = val_auc
                torch.save(model.state_dict(), best_path)
                logging.info(f"Saved best model -> {best_path}")

    # final testing
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=torch.device("cpu")))
    test_auc, test_loss, test_report = evaluate(model, data, mask_test, loss_fn, label_map)

    # save metrics and label_map
    metrics = {"test_auc": float(test_auc) if not np.isnan(test_auc) else None,
               "best_val_auc": float(best_val_auc) if not np.isnan(best_val_auc) else None,
               "label_map": label_map}
    with open(os.path.join(args.save_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[FINAL TEST]")
    print("Test AUC:", test_auc)
    print(test_report)
# ============================================================
# ------------------------ CLI  ------------------------------
# ============================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()

    # dataset paths
    p.add_argument("--dataset_dir", type=str, default="./dataset")
    p.add_argument("--addr_csv", type=str, default="./dataset/synthetic_ethereum_multiclass_dataset_v2.csv")
    p.add_argument("--edges_csv", type=str, default="./dataset/synthetic_ethereum_edges_v2.csv")

    # output
    p.add_argument("--save_dir", type=str, default="./model2_out")

    # model sizes
    p.add_argument("--hidden_gnn", type=int, default=64)
    p.add_argument("--hidden_trans", type=int, default=64)
    p.add_argument("--gnn_heads", type=int, default=2)
    p.add_argument("--trans_heads", type=int, default=4)
    p.add_argument("--trans_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)

    # temporal
    p.add_argument("--time_bins", type=int, default=32)

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_every", type=int, default=5)

    # mixup
    p.add_argument("--mixup_alpha", type=float, default=0.4)

    # focal loss
    p.add_argument("--focal_gamma", type=float, default=2.0)

    # dynamic class weighting
    p.add_argument("--weight_blend", type=float, default=0.5)

    # misc
    p.add_argument("--clip_grad_norm", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    main_train(args)
