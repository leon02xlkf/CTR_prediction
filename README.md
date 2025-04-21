# CTR_prediction

## Project Overview
This project implements an end-to-end news recommendation system that leverages users' historical click behavior and article content features. The pipeline consists of two main stages:

1. **Recall**: Quickly retrieve a candidate set of news items from a large corpus.
2. **Ranking**: Fine-tune and rank the candidate set for precise personalization.

Key techniques include multi-channel recall (ItemCF, UserCF, YouTubeDNN), dual-tower self-supervised pretraining, and advanced ranking models such as LightGBM and Deep Interest Network (DIN).


## Environment Setup
Create and activate a conda environment, then install dependencies:
```bash
conda create -n newsrec python=3.8 -y
conda activate newsrec
pip install -r requirements.txt
```

**requirements.txt** includes:
- `numpy`, `pandas`, `scikit-learn`   — data processing and feature engineering
- `lightgbm`                         — gradient boosting ranking model
- `torch`, `torchvision`             — deep learning frameworks for DNN and DIN
- `faiss`                            — fast vector retrieval
- `matplotlib`, `seaborn`            — visualization utilities

## Data Preparation
1. **Raw Log Cleaning**: Deduplicate events, remove invalid records, normalize timestamps.
2. **Train/Test Split**: Hold out each user’s last click as the test set; use earlier clicks for training.
3. **Feature Engineering**:
   - **User-side**: historical click counts, time-window statistics, TF-IDF text features.
   - **Item-side**: category one-hot, word embeddings from headlines, image features via pre-trained CNN.
4. **Self-Supervised Samples**: Uniformly sample articles and apply random masking to construct contrastive views.

Run preprocessing:
```bash
python src/utils/prepare_data.py
```

## Recall Module
- **ItemCF / UserCF**: Collaborative filtering based on co-click statistics.  
- **YouTubeDNN**: Dual-tower architecture with user and item sub-networks, paired by vector dot-product.  
- **Ensembling**: Merge multiple recall results with weighted scores and select Top-K candidates.

Example usage:
```python
from src.recall.recommend import recall_for_user
candidates = recall_for_user(user_id=12345, topk=100)
```

## Ranking Module
- **LightGBM**: Gradient boosting decision trees for candidate re-scoring.  
- **DIN (Deep Interest Network)**: Attention-based sequence model capturing short-term user interest.

Training commands:
```bash
python src/ranking/train_lgbm.py --config configs/lgbm.yaml
python src/ranking/train_din.py  --config configs/din.yaml
```

Inference:
```bash
python src/ranking/predict.py --model lgbm --user 12345
```

## Self-Supervised Pretraining
We apply random feature masking and contrastive loss to pretrain the dual-tower encoders, improving representation robustness.
```bash
python src/self_supervised/train_masked_tower.py
```

## Evaluation Metrics
- **Recall@K** for recall quality.  
- **NDCG@K**, **MAP@K** for ranking performance.  

Detailed experiments and visualizations are available in `notebooks/`.

## Results
- **Offline**: Recall@100 = 0.75, NDCG@10 = 0.32.  
- **Online A/B Test**: 12% increase in click-through rate compared to baseline.

## Future Work
- Integrate reinforcement learning for long-term user satisfaction.  
- Incorporate real-time streaming features (Flink/Spark Streaming).  
- Explore multi-modal fusion with knowledge graphs and user profiles.

---
**Maintainer**: Your Name (<you@example.com>)  
**License**: MIT License

