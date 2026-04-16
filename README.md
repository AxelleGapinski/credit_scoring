---
title: Credit scoring
emoji: 💵
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.4.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

## Projet : Déploiement et monitoring d’un modèle de scoring crédit

Mise en production d'un modèle de scoring crédit développé dans une première phase (notebooks) et fournissant:
- une API de scoring,
- une conteneurisation Docker,
- un pipeline CI/CD,
- un monitoring des prédictions et des performances
Les données de production sont simulées à partir d’un sous‑ensemble du jeu d’entraînement.

## Architecture du repo
```bash
├── app.py                      # API de scoring (Gradio)
├── model.pkl                   # Modèle entraîné (LightGBM)
├── requirements.txt
├── Dockerfile
├── README.md
├── train_test/
│   └── sample_train.csv        # Sous-ensemble du dataset train d’origine
├── logs/
│   └── predictions.jsonl       # Logs des appels API
├── monitoring/
│   ├── analyse_logs.py         # Analyse des logs (métriques, drift)
│   └── dashboard_logs.py       # Dashboard Streamlit de monitoring
├── tests/
│   └── test_api.py             # Tests automatisés de l’API
├── notebooks/
│   ├── 1_preparation_donnees.ipynb
│   ├── 2_3_mlflow_modelisation.ipynb
│   └── 4_optimisation1.ipynb
└── .github/workflows/
    ├── ci.yml                  # CI : tests + build Docker
    └── sync.yml                # Sync vers Hugging Face Spaces
```

## Lancer l’API localement

1. Installer les dépendances
```bash
pip install -r requirements.txt
```
2. Lancer l’application
```bash
python app.py
```

API  accessible sur : http://localhost:7860
