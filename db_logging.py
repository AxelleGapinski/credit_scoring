import os
import json
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("NEON_DATABASE_URL")

def get_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    '''
    Crée la table predictions dans NeonDB
    '''
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              SERIAL PRIMARY KEY,
            timestamp       TIMESTAMP DEFAULT NOW(),
            client_id       INTEGER,
            prediction      TEXT,
            prediction_proba FLOAT,
            latency_ms      FLOAT,
            status          TEXT,
            input_features  JSONB,
            timings         JSONB
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Table predictions crééé")

def log_prediction(client_id, prediction, prediction_proba, latency_ms, status, input_features=None, timings=None, error_message=None):
    '''
    Ajoute une prédiction dans NeonDB
    Args:
        client_id : ID du client
        prediction : "Crédit à accorder" ou "Crédit à refuser"
        prediction_proba : probabilité de ne pas rembourser
        latency_ms : temps de réponse total en ms
        status : "success" ou "error"
        input_features : dict des features du client 
        timings : dict des temps par étape
    '''
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions
                (client_id, prediction, prediction_proba, latency_ms, status, input_features, timings)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            client_id,
            prediction,
            prediction_proba,
            latency_ms,
            status,
            json.dumps(input_features) if input_features else None,
            json.dumps(timings) if timings else None,
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Erreur log NeonDB : {e}")


def fetch_predictions(limit=3000):
    """
    Récupère les dernières prédictions depuis NeonDB (utilisé par le dashboard et le notebook de drift)
    """
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT * FROM predictions
        ORDER BY timestamp DESC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows