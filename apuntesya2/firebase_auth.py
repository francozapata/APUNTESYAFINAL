import os, json
import firebase_admin
from firebase_admin import credentials, auth as fb_auth

_cred = None
_initialized = False

def init_firebase_admin():
    global _cred, _initialized
    if _initialized:
        return
    sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if not sa_json:
        print("[Firebase] WARN: FIREBASE_SERVICE_ACCOUNT_JSON no configurado. Verificación de token no disponible.")
        return
    try:
        # If the env var contains JSON with escaped quotes/newlines, try to load robustly
        data = json.loads(sa_json)
        _cred = credentials.Certificate(data)
        firebase_admin.initialize_app(_cred, {
            "projectId": data.get("project_id")
        })
        _initialized = True
        print("[Firebase] Admin SDK inicializado.")
    except Exception as e:
        print("[Firebase] ERROR inicializando Admin SDK:", e)

def verify_id_token(id_token: str):
    """Devuelve el dict del token verificado o None si falla."""
    if not firebase_admin._apps:
        init_firebase_admin()
    if not firebase_admin._apps:
        return None
    try:
        decoded = fb_auth.verify_id_token(id_token, clock_skew_seconds=60)
        return decoded
    except Exception as e:
        print("[Firebase] verify_id_token error:", e)
        return None
