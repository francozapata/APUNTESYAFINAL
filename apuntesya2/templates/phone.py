# app/utils/phone.py
import re

COUNTRY_DEFAULT = '+54'  # Argentina por defecto

_digits = re.compile(r'\D+')

def _only_digits(s: str) -> str:
    return _digits.sub('', s or '')

def normalize_phone_e164(raw: str | None) -> str | None:
    """
    Convierte distintas formas de teléfono a algo cercano a E.164.
    Si no trae +54, se lo agregamos por defecto (ajustá COUNTRY_DEFAULT si querés).
    """
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith('+'):
        return '+' + _only_digits(raw)
    digits = _only_digits(raw)
    if not digits:
        return None
    if digits.startswith('0'):
        digits = digits[1:]
    return COUNTRY_DEFAULT + digits

def whatsapp_link(e164: str | None) -> str | None:
    if not e164:
        return None
    return f"https://wa.me/{e164.replace('+','')}"
