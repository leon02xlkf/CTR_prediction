# CTR_prediction

## Project Overview
This project implements an end-to-end news recommendation system that leverages users' historical click behavior and article content features. The pipeline consists of two main stages:

1. **Recall**: Quickly retrieve a candidate set of news items from a large corpus.
2. **Ranking**: Fine-tune and rank the candidate set for precise personalization.

Key techniques include multi-channel recall (ItemCF, UserCF, YouTubeDNN), dual-tower self-supervised pretraining, and advanced ranking models such as LightGBM and Deep Interest Network (DIN).
