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
- un monitoring des prédictions et des performances,
- un notebook d'analyse du data drift, 
- une base de données NeonDB pour stoker les logs de prédictions
Les données de production sont simulées à partir d’un sous‑ensemble du jeu d’entraînement.

## Architecture du repo
```bash
├── app.py                      # API de scoring (Gradio)
├── model.pkl                   # Modèle entraîné (LightGBM)
├── requirements.txt
├── Dockerfile
├── README.md
├── dashboard_logs.py       # Dashboard Streamlit de monitoring
├── data_drift_analysis_ipynb  # notebook d'analyse du data drift
├── db_logging.py  # gestion de la base de données NeonDB et des logs de prédiction stockés
├── train_test/
│   └── sample_train.csv        # Sous-ensemble du dataset train d’origine(métriques, drift)
├── tests/
│   └── test_api.py             # Tests automatisés de l’API
├── notebooks/
│   ├── 1_preparation_donnees.ipynb
│   ├── 2_3_mlflow_modelisation.ipynb
│   └── 4_optimisation1.ipynb
└── .github/workflows/
    └── ci_cd.yml                  # CI : tests + build Docker et CD sync vers HF Spaces
```

## Monitoring

### Lancer le dashboard
```bash
streamlit run monitoring/dashboard_logs.py
```
Dashboard accessible sur : http://localhost:8501

### Ce que monitore le dashboard
- Métriques opérationnelles : latence moyenne, taux d'erreurs, nombre de requêtes
- Distribution des décisions : proportion de crédits accordés/refusés
- Drift des variables: comparaison des distributions entre train et production pour quelques variables


### Base de données de production
Les prédictions sont stockées dans NeonDB (PostgreSQL cloud).
Chaque appel API enregistre : timestamp, ID client, score, décision, latence, features.

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
