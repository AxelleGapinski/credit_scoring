import streamlit as st
import pandas as pd
import json

ref_data = pd.read_csv('./train_test/sample_train.csv')

## 
# Dashboard streamlit d'analyse des logs de prédiction 
##

st.title("Monitoring des logs de credit scoring")

# charger les logs
try:
    records = [json.loads(line) for line in open("logs/predictions.jsonl")]
    df = pd.DataFrame(records)
except FileNotFoundError:
    st.error("Fichier de logs introuvable")
    st.stop()

# mettre timings en colonnes
timings_df = df['timings'].apply(pd.Series)
timings_df.columns = [f"timing_{col}" for col in timings_df.columns]
df = pd.concat([df, timings_df], axis=1)

# nettoyer les données
df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
df['status'] = df['status'].astype(str)
df['prediction'] = df['prediction'].fillna("None")

# Métriques globales
st.subheader("Métriques globales")
st.write("Total de requêtes:", len(df))
st.write("Taux d'erreurs:", f"{(df['status'] != 'success').mean() * 100:.2f} %")
st.write("Latence moyenne:", f"{df['latency_ms'].mean():.2f} ms")
st.write("Latence max :", f"{df['latency_ms'].max():.2f} ms")

# distribution latence
st.subheader("Distribution de la latence")
latency_hist = pd.cut(df['latency_ms'], bins=20).value_counts().sort_index()
latency_hist.index = latency_hist.index.astype(str)
st.bar_chart(latency_hist)

# distribution des prédictions
st.subheader("Répartition des décisions du modèle")
decision_counts = df['prediction'].value_counts()
st.bar_chart(decision_counts)

# Drift des variables
st.subheader("Drift des variables")

features_df = df['input_features'].apply(pd.Series)
features_df.columns = [f"feat_{col}" for col in features_df.columns]
df = pd.concat([df, features_df], axis=1)

drift_results = []
for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'CNT_CHILDREN', 'AMT_ANNUITY', 'NAME_CONTRACT_TYPE']:
    feat_col = f"feat_{col}"
    if feat_col in df.columns and col in ref_data.columns:
        if df[feat_col].notna().sum() > 0:
            ref_mean = ref_data[col].mean()
            prod_mean = df[feat_col].mean()
            variation = abs(ref_mean - prod_mean) / (abs(ref_mean) + 1e-6)

            drift_results.append({
                "feature": col,
                "ref_mean": ref_mean,
                "prod_mean": prod_mean,
                "variation": variation
            })

drift_df = pd.DataFrame(drift_results)
drift_df["variation_pct"] = (drift_df["variation"] * 100).round(2)

# seuil de drift 
threshold = 0.1

# flag alerte drift
drift_df["drift_detected"] = drift_df["variation"] > threshold

st.dataframe(drift_df[["feature", "ref_mean", "prod_mean", "variation_pct", "drift_detected"]])

global_drift = drift_df["drift_detected"].any()
if global_drift:
    st.error("Drift détecté sur certaines variables")
else:
    st.success("Aucun drift détecté")

# Statut des requêtes erreur ou success
st.subheader("Répartition requêtes erreurs/succès")
status_counts = df['status'].value_counts()
st.bar_chart(status_counts)

# dernières requêtes faites
st.subheader("Dernières requêtes")
st.dataframe(df[['timestamp', 'client_id', 'prediction','prediction_proba', 'latency_ms', 'status']].tail(20))

# timing moyen par étape de requête
st.subheader("Temps moyen par étape")
mean_timings = df[['timing_validation', 'timing_search_client', 'timing_get_client_row', 'timing_extract_client_features', 'timing_prediction', 'timing_total']].mean()
mean_timings.index = [c.replace('timing_', '') for c in mean_timings.index]

st.bar_chart(mean_timings)

