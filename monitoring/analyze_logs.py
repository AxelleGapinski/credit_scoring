import json
import os
import pandas as pd
import numpy as np

log_file = "logs/predictions.jsonl"
ref_data = "./train_test/sample_train.csv"

def clean_col_names(df):
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df

def load_logs(log_file=log_file):
    if not os.path.exists(log_file):
        print("aucun fichier de log trouvé")
        return pd.DataFrame()

    records = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("Fichier de log vide")
        return pd.DataFrame()

    return pd.DataFrame(records)

def extract_production_features(df_logs):
    """
    Transforme la colonne input_features en df exploitable
    """
    success_df = df_logs[df_logs["status"] == "success"].copy()

    if success_df.empty or "input_features" not in success_df.columns:
        return pd.DataFrame()

    feature_rows = []
    for _, row in success_df.iterrows():
        features = row.get("input_features", {})
        if isinstance(features, dict) and features:
            feature_rows.append(features)

    if not feature_rows:
        return pd.DataFrame()

    return pd.DataFrame(feature_rows)

def compute_operational_metrics(df):
    total_requests = len(df)
    error_rate = (df["status"] == "error").mean() if total_requests > 0 else 0
    success_df = df[df["status"] == "success"].copy()

    if not success_df.empty and "latency_ms" in success_df.columns:
        mean_latency = success_df["latency_ms"].mean()
        p95_latency = success_df["latency_ms"].quantile(0.95)
        max_latency = success_df["latency_ms"].max()
    else:
        mean_latency = p95_latency = max_latency = np.nan

    print("--- Métriques ---")
    print(f"Total de requêtes : {total_requests}")
    print(f"Taux d'erreurs: {error_rate:.2%}")
    print(f"Latence moyenne : {mean_latency:.2f} ms" if not np.isnan(mean_latency) else "Latence moyenne: N/A")
    print(f"Latence max: {max_latency:.2f} ms" if not np.isnan(max_latency) else "Latence max :N/A")
    print()

    return {
        "total_requests": total_requests,
        "error_rate": error_rate,
        "mean_latency": mean_latency,
        "p95_latency": p95_latency,
        "max_latency": max_latency
    }

def detect_operational_anomalies(metrics):
    print("-- Anomalies --")
    alerts = []

    if metrics["error_rate"] > 0.05:
        alerts.append("Taux d'erreur de plus de 5%")

    if not alerts:
        print("Pas d'anomalie détectée")
    else:
        for alert in alerts:
            print(alert)

    print()
    return alerts

def analyze_prediction_distribution(df):
    print("--- Distribution des prédictions ---")

    success_df = df[df["status"] == "success"].copy()
    if success_df.empty or "prediction" not in success_df.columns:
        print("Pas de prédictions exploitable dans les logs")
        print()
        return

    counts = success_df["prediction"].value_counts(normalize=True) * 100
    print(counts.round(2).astype(str) + " %")
    print()

def load_reference_features():
    if not os.path.exists(ref_data):
        print("Fichier de référence introuvable")
        return pd.DataFrame()

    ref_df = pd.read_csv(ref_data)
    # même nettoyage que l'app
    ref_df.columns = [__import__("re").sub(r'[^A-Za-z0-9_]', '_', col) for col in ref_df.columns]

    # garder seulement les features du modèle
    ref_df = ref_df.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")
    return ref_df

def simple_drift_analysis(ref_df, prod_df, max_features=10):
    """
    Drift simple: compare moyenne relative sur variables numériques
    Alerte si variation de plus de 20%
    Drift = quand les données en prod ne ressemblent plus à ce que le modèle a vu en entraînement = risque de dégradation des performances
    """
    print("--- Analyse de drift ---")

    if ref_df.empty:
        print("data de référence indisponible")
        print()
        return

    if prod_df.empty:
        print("Pas de données dans les logs")
        print()
        return

    common_cols = [c for c in ref_df.columns if c in prod_df.columns]
    if not common_cols:
        print("Pas de feature commune entre data de référence et de production")
        print()
        return

    numeric_cols = []
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(ref_df[col]) and pd.api.types.is_numeric_dtype(prod_df[col]):
            numeric_cols.append(col)

    if not numeric_cols:
        print("Pas de variable numérique exploitable pour le drift")
        print()
        return

    drifted_features = []

    for col in numeric_cols[:max_features]:
        ref_mean = ref_df[col].dropna().mean()
        prod_mean = prod_df[col].dropna().mean()

        if pd.isna(ref_mean) or pd.isna(prod_mean):
            continue

        if abs(ref_mean) < 1e-9:
            diff_ratio = abs(prod_mean - ref_mean)
        else:
            diff_ratio = abs(prod_mean - ref_mean) / abs(ref_mean)

        print(
            f"{col}: ref_mean={ref_mean:.4f} - prod_mean={prod_mean:.4f} - variation={diff_ratio:.2%}"
        )

        if diff_ratio > 0.20:
            drifted_features.append(col)

    print()

    if drifted_features:
        print("alerte drift détecté sur :", ", ".join(drifted_features))
    else:
        print("Aucun drift important trouvé dans les variables analysées")

    print()
    return drifted_features

def main():
    df_logs = load_logs()
    if df_logs.empty:
        return

    metrics = compute_operational_metrics(df_logs)
    detect_operational_anomalies(metrics)
    analyze_prediction_distribution(df_logs)

    prod_features_df = extract_production_features(df_logs)
    ref_features_df = load_reference_features()

    simple_drift_analysis(ref_features_df, prod_features_df, max_features=10)

if __name__ == "__main__":
    main()