# Hybrid GNN-Transformer Blockchain Fraud Detection

A research-oriented blockchain fraud detection framework that combines **Graph Neural Networks (GNNs)** and **Transformer architectures** to identify fraudulent Ethereum transactions through graph representation learning, temporal transaction modeling, and explainable artificial intelligence (XAI).

---

## 📌 Overview

Blockchain technology has transformed digital finance by enabling decentralized and transparent transactions. However, the increasing adoption of blockchain networks has also led to sophisticated fraudulent activities such as phishing, Ponzi schemes, rug pulls, money laundering, and malicious smart contracts.

This project presents a **Hybrid GNN-Transformer** framework that models blockchain transactions as graphs while capturing temporal transaction behavior to accurately detect fraudulent activities in real time.

---

## ✨ Features

- Graph-based Ethereum transaction modeling
- Hybrid Graph Neural Network + Transformer architecture
- Temporal transaction sequence learning
- Real-time fraud prediction
- Explainable AI (XAI) support
- Interactive analytics dashboard
- Multi-class fraud detection
- Scalable preprocessing pipeline
- Model evaluation and benchmarking

---

## 🏗️ System Architecture

```
Ethereum Transaction Data
          │
          ▼
 Data Collection & Integration
          │
          ▼
 Data Cleaning & Feature Engineering
          │
          ▼
 Graph Construction
(Transaction Graph)
          │
          ▼
 Hybrid GNN Encoder
          │
          ▼
 Transformer Encoder
          │
          ▼
 Fraud Classification
          │
          ▼
 Explainable AI
          │
          ▼
 Analytics Dashboard
```

---

## 🚀 Technologies Used

### Programming

- Python

### Machine Learning

- PyTorch
- PyTorch Geometric
- Scikit-learn
- NumPy
- Pandas

### Deep Learning

- Graph Neural Networks (GAT/GCN)
- Transformer Networks
- Attention Mechanisms

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Dashboard

- Streamlit

### Data Sources

- Ethereum Blockchain
- Etherscan API
- CryptoScamDB
- PhishFort

---

## 📂 Project Structure

```
Hybrid-GNN-Transformer-Blockchain-Fraud/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── graph_data/
│
├── models/
│   ├── gnn.py
│   ├── transformer.py
│   ├── hybrid_model.py
│   └── train.py
│
├── preprocessing/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── graph_builder.py
│
├── dashboard/
│   ├── app.py
│   └── visualization.py
│
├── explainability/
│   ├── shap_analysis.py
│   └── gnn_explainer.py
│
├── utils/
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

## 📊 Fraud Categories

The framework is designed to identify multiple blockchain fraud types including:

- Ponzi Schemes
- Phishing Attacks
- Rug Pulls
- Malicious Smart Contracts
- Scam Wallets
- Money Laundering Activities
- Suspicious Transactions
- Legitimate Transactions

---

## 🧠 Model Pipeline

1. Collect blockchain transaction data
2. Clean and preprocess datasets
3. Generate graph representations
4. Extract graph and temporal features
5. Train Hybrid GNN-Transformer model
6. Predict fraudulent transactions
7. Generate explainable predictions
8. Visualize results through dashboard

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Precision-Recall Curve

---

## 💡 Future Improvements

- Graph Transformer Networks
- Contrastive Graph Learning
- Federated Fraud Detection
- Cross-chain Fraud Detection
- Real-time Blockchain Monitoring
- Large Language Model Integration
- Adaptive Fraud Detection
- Knowledge Graph Integration

---

## 🎯 Applications

- Cryptocurrency Exchanges
- Financial Institutions
- Blockchain Security
- AML (Anti-Money Laundering)
- Regulatory Compliance
- Web3 Security Platforms
- DeFi Protocol Monitoring
- Smart Contract Auditing

---

## 📌 Research Contribution

This project proposes a hybrid deep learning framework that combines:

- Graph representation learning
- Temporal sequence modeling
- Attention mechanisms
- Explainable AI
- Real-time blockchain analytics

The integration of GNNs with Transformers enables improved fraud detection performance by capturing both structural relationships and temporal transaction patterns within blockchain networks.

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/hybrid-gnn-transformer-blockchain-fraud.git
```

Move into the project directory

```bash
cd hybrid-gnn-transformer-blockchain-fraud
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the training pipeline

```bash
python train.py
```

Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🤝 Contributing

Contributions are welcome through pull requests. Please open an issue first to discuss significant changes or feature additions.

---

## 📄 License

This project is intended for educational and research purposes.

---

## 👨‍💻 Author

**Varad Parkhe**

M.Tech – Data Science & Analytics  
MIT World Peace University



---

## ⭐ Acknowledgements

- PyTorch
- PyTorch Geometric
- Scikit-learn
- Ethereum Foundation
- Etherscan
- Open Source Community

---

**If you find this project useful, consider giving it a ⭐ on GitHub.**
