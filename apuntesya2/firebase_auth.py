import firebase_admin
from firebase_admin import auth as fb_auth

def verify_id_token(id_token: str):
    """Devuelve el dict del token verificado o None si falla."""
    if not firebase_admin._apps:
        print("[Firebase] ERROR: Firebase Admin no inicializado (verify_id_token)")
        return None

    try:
        decoded = fb_auth.verify_id_token(id_token, clock_skew_seconds=60)
        return decoded
    except Exception as e:
        print("[Firebase] verify_id_token error:", e)
        return None