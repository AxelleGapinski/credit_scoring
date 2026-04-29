import streamlit as st
import pandas as pd
import json
from db_logging import fetch_predictions

ref_data = pd.read_csv('./train_test/sample_train.csv')

st.title("Monitoring des logs de credit scoring")

# charger les logs depuis NeonDB
try:
    records = fetch_predictions(limit=3000)
    df = pd.DataFrame(records)
    # input_features est stocké en JSONB — le convertir en dict si nécessaire
    df['input_features'] = df['input_features'].apply(
        lambda x: x if isinstance(x, dict) else json.loads(x) if x else {}
    )
    df['timings'] = df['timings'].apply(
        lambda x: x if isinstance(x, dict) else json.loads(x) if x else {}
    )
except Exception as e:
    st.error(f"Erreur chargement NeonDB: {e}")
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

