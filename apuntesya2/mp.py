import os, requests
from datetime import datetime, timedelta

API_BASE = "https://api.mercadopago.com"

def _auth_header(access_token:str):
    return {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

def oauth_authorize_url():
    client_id = os.getenv("MP_OAUTH_CLIENT_ID")
    redirect_uri = os.getenv("MP_OAUTH_REDIRECT_URL")
    return f"https://auth.mercadopago.com/authorization?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}"

def oauth_exchange_code(code:str):
    data = {
        "grant_type": "authorization_code",
        "client_id": os.getenv("MP_OAUTH_CLIENT_ID"),
        "client_secret": os.getenv("MP_OAUTH_CLIENT_SECRET"),
        "code": code,
        "redirect_uri": os.getenv("MP_OAUTH_REDIRECT_URL"),
    }
    r = requests.post(f"{API_BASE}/oauth/token", data=data, timeout=30)
    r.raise_for_status()
    return r.json()

def oauth_refresh(refresh_token:str):
    data = {
        "grant_type": "refresh_token",
        "client_id": os.getenv("MP_OAUTH_CLIENT_ID"),
        "client_secret": os.getenv("MP_OAUTH_CLIENT_SECRET"),
        "refresh_token": refresh_token
    }
    r = requests.post(f"{API_BASE}/oauth/token", data=data, timeout=30)
    r.raise_for_status()
    return r.json()

def create_preference_for_seller_token(seller_access_token:str, title:str, unit_price:float, quantity:int, marketplace_fee:float, external_reference:str, back_urls:dict, notification_url:str):
    # back_urls sanity
    back_urls = back_urls or {}
    for k in ("success","failure","pending"):
        if k in back_urls and not isinstance(back_urls[k], str):
            back_urls[k] = str(back_urls[k])

    payload = {
        "items":[{
            "title": title,
            "quantity": int(quantity),
            "currency_id": "ARS",
            "unit_price": float(unit_price)
        }],
        "external_reference": external_reference,
        "back_urls": back_urls,
        "notification_url": notification_url,
        "marketplace_fee": round(float(marketplace_fee), 2)
    }

    # Activar auto_return SOLO si success es https
    success_url = back_urls.get("success", "")
    if isinstance(success_url, str) and success_url.startswith("https://"):
        payload["auto_return"] = "approved"

    r = requests.post(f"{API_BASE}/checkout/preferences", json=payload, headers=_auth_header(seller_access_token), timeout=30)
    # Mejor diagnóstico
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"raw": r.text}
        raise RuntimeError(f"MP preference error {r.status_code}: {err}")
    try:
        return r.json()
    except Exception as e:
        # Evita "'Response' object is not callable" por confusiones al serializar
        raise RuntimeError(f"No se pudo parsear la respuesta de MP: {e}; cuerpo={r.text[:400]}")

def get_payment(access_token:str, payment_id:str):
    r = requests.get(f"{API_BASE}/v1/payments/{payment_id}", headers=_auth_header(access_token), timeout=30)
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"raw": r.text}
        raise RuntimeError(f"MP get_payment error {r.status_code}: {err}")
    return r.json()


def search_payments_by_external_reference(access_token:str, external_reference:str):
    params = {"external_reference": external_reference, "sort": "date_created", "criteria": "desc"}
    r = requests.get(f"{API_BASE}/v1/payments/search", params=params, headers=_auth_header(access_token), timeout=30)
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"raw": r.text}
        raise RuntimeError(f"MP search error {r.status_code}: {err}")
    return r.json()
