from app import predict

def test_predict_existing_client():
    """
    Vérifie que le résultat pour un client qui existe retourne bien une probabilité et une décision
    """
    proba, decision = predict(100002)
    assert isinstance(proba, str)
    assert isinstance(decision, str)
    assert "%" in proba
    assert decision in ["Crédit à refuser", "Crédit à accorder"]


def test_predict_non_existing_client():
    """
    Vérifie qu'un ID de client qui n'existe pas retourne une erreur
    """
    proba, decision = predict(999999999)
    assert proba == "Client introuvable"
    assert decision == ""


def test_predict_invalid_id_string():
    """
    Vérifie qu'un ID invalide (texte par ex) retourne une erreur propre.
    """
    proba, decision = predict("abcdefg")
    assert proba == "ID invalide"
    assert decision == ""


def test_predict_invalid_id_none():
    """
    Vérifie qu'un ID qui est None retourne une erreur
    """
    proba, decision = predict(None)
    assert proba == "ID invalide"
    assert decision == ""


def test_predict_output_length():
    """
    Vérifie que predict retourne bien 2 valeurs
    """
    result = predict(100002)
    assert isinstance(result, tuple)
    assert len(result) == 2