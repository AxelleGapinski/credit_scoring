import gradio as gr
import joblib
import pandas as pd
import re
# pour log infos ####----
import os
import json
import time
import uuid
from datetime import datetime
####----

# chargement du modèle et des données
model = joblib.load('model.pkl')
data = pd.read_csv('./train_test/sample_train.csv')

THRESHOLD = 0.46  # meilleur seuil

####----
# Logging local
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "predictions.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

def log_prediction(
    client_id,
    input_features=None,
    prediction=None,
    prediction_proba=None,
    latency_ms=None,
    status="success",
    error_message=None
):
    """
    Enregistre un appel de prédiction dans un fichier JSONL.
    """
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "request_id": str(uuid.uuid4()),
        "client_id": client_id,
        "input_features": input_features if input_features is not None else {},
        "prediction": prediction,
        "prediction_proba": prediction_proba,
        "latency_ms": latency_ms,
        "status": status,
        "error_message": error_message
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

####----

def clean_col_names(df):
    df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]
    return df

data = clean_col_names(data)

####----
# Colonnes utilisées pour le modèle (sans ID ni target)
FEATURE_COLUMNS = [col for col in data.columns if col not in ["SK_ID_CURR", "TARGET"]]

def make_json_serializable(d):
    """
    Convertit les types numpy/pandas en types Python simples pour JSON.
    """
    clean_dict = {}
    for k, v in d.items():
        if pd.isna(v):
            clean_dict[k] = None
        elif hasattr(v, "item"):
            clean_dict[k] = v.item()
        else:
            clean_dict[k] = v
    return clean_dict

###----

# Prédiction
# =========================
def predict(client_id: int):
    """
    Prend un ID de client (SK_ID_CURR), récupère les features du client,
    retourne la probabilité de défaut et la décision.
    Logge aussi les appels pour monitoring / drift.
    """
    start_time = time.time()

    # check si l'ID est en format valide
    try:
        client_id = int(client_id)
    except (ValueError, TypeError):
        latency_ms = (time.time() - start_time) * 1000
        log_prediction(
            client_id=client_id,
            input_features={},
            prediction=None,
            prediction_proba=None,
            latency_ms=latency_ms,
            status="error",
            error_message="ID invalide"
        )
        return "ID invalide", ""

    # check si le client existe
    if client_id not in data['SK_ID_CURR'].values:
        latency_ms = (time.time() - start_time) * 1000
        log_prediction(
            client_id=client_id,
            input_features={},
            prediction=None,
            prediction_proba=None,
            latency_ms=latency_ms,
            status="error",
            error_message="Client introuvable"
        )
        return "Client introuvable", ""

    # récupérer la ligne complète du client
    client_row = data[data['SK_ID_CURR'] == client_id]

    # récupérer les features du client
    client = client_row[FEATURE_COLUMNS]

    # features brutes pour logging / drift
    input_features_dict = make_json_serializable(client.iloc[0].to_dict())

    # prédiction
    proba = model.predict_proba(client)[0][1]
    decision = "Crédit à refuser" if proba >= THRESHOLD else "Crédit à accorder"

    latency_ms = (time.time() - start_time) * 1000

    # log succès
    log_prediction(
        client_id=client_id,
        input_features=input_features_dict,
        prediction=decision,
        prediction_proba=float(proba),
        latency_ms=latency_ms,
        status="success",
        error_message=None
    )

    return (
        f"{proba:.2%}",
        decision
    )
#-----

# Interface Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Number(label="ID Client"),
    outputs=[
        gr.Text(label="Probabilité de ne pas rembourser le crédit"),
        gr.Text(label="Décision")
    ],
    title="Scoring crédit",
    description="Entrez l'ID d'un client pour obtenir son score de risque de ne pas rembourser et la décision d'attribution du crédit",
    examples=[[100002], [100055], [456254]],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)