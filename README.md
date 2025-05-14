## 📌 Project Overview

**MBFT-Lite** (*Modified Multi-Branch Federated Transformer*) is a **privacy-preserving phishing email detection framework** that combines the power of deep learning and federated learning.

The system integrates:

- **TinyBERT** for semantic understanding of email content
- **XGBoost** for metadata-based structural analysis
- **Federated Learning (FL)** to enable decentralized training across clients
- **Fusion strategies** including:
  - Weighted Averaging
  - Max Confidence Voting
  - Logistic Regression-based Meta Classifier

MBFT-Lite is trained and evaluated on **real-world phishing and legitimate email datasets**, simulating four distributed clients with heterogeneous data. The framework ensures that **no raw email data is shared**, supporting **data privacy, generalization, and scalability**.

> ✅ Designed for real-world deployment across industries such as **finance, education, and healthcare**.
