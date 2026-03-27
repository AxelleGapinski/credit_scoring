import gradio as gr
import joblib
import pandas as pd
import re

# chargement du modèle et des données
model = joblib.load('model.pkl')
data = pd.read_csv('C:/Users/axell/Documents/projets_code/projet_6/projet6/train_test/train_final.csv')

THRESHOLD = 0.46  # meilleur seuil

def clean_col_names(df):
    df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]
    return df

data = clean_col_names(data)

def predict(client_id: int):
    """
    Prend un ID de client (SK_ID_CURR), récupère les features du client, retourne la probabilité de défaut et la décision
    """

    # check si l'ID est en format valide
    try:
        client_id = int(client_id)
    except (ValueError, TypeError):
        return "ID invalide", ""

    # check si le client existe
    if client_id not in data['SK_ID_CURR'].values:
        return "Client introuvable", ""

    # récupérer les features du client
    client = data[data['SK_ID_CURR'] == client_id].drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')

    # prédiction
    proba = model.predict_proba(client)[0][1]
    decision = "Crédit à refuser" if proba >= THRESHOLD else "Crédit à accorder" # credit refusé si proba + grande que le seuil optimal

    return (
        f"{proba:.2%}",
        decision
    )

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