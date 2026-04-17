import streamlit as st
import pandas as pd
import json

##
# Dashboard streamlit d'analyse des logs de prédiction
##

st.title("Monitoring des logs de credit scoring")

# charger les logs
try:
    records = [json.loads(line) for line in open("logs/predictions.jsonl")]
    df = pd.DataFrame(records)
except FileNotFoundError:
    st.error()
    st.stop()

# nettoyer les données
df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
df['status'] = df['status'].astype(str)
df['prediction'] = df['prediction'].fillna("None")

# Mmétriques globales
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

# clients avec + grandes latences
st.subheader("Clients avec les plus fortes latences")
top_latency = df.sort_values('latency_ms', ascending=False).head(10)
st.dataframe(
    top_latency[['timestamp','client_id','latency_ms','status']]
)
# Distribution prédictions
st.subheader("Distribution des prédictions")
pred_counts = df['prediction'].value_counts()
st.bar_chart(pred_counts)

# drift des prédictions
st.subheader("Drift des prédictions")
pred_ratio = df['prediction'].value_counts(normalize=True)
st.bar_chart(pred_ratio)

# Statut des requêtes erreur ou success
st.subheader("Répartition requêteserreurs/succès")
status_counts = df['status'].value_counts()
st.bar_chart(status_counts)

# dernières requêtes faites
st.subheader("Dernières requêtes")
st.dataframe(df[['timestamp','client_id','prediction','prediction_proba','latency_ms','status']].tail(20))