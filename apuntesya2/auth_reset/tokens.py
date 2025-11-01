from itsdangerous import URLSafeTimedSerializer
from flask import current_app

def _serializer():
    return URLSafeTimedSerializer(current_app.config['SECRET_KEY'])

def generate_token(email: str) -> str:
    return _serializer().dumps(email, salt=current_app.config.get('SECURITY_PASSWORD_SALT', 'pw-reset'))

def confirm_token(token: str, expiration: int = None):
    if expiration is None:
        expiration = int(current_app.config.get('PASSWORD_RESET_EXPIRATION', 3600))
    try:
        email = _serializer().loads(token, salt=current_app.config.get('SECURITY_PASSWORD_SALT', 'pw-reset'), max_age=expiration)
        return email
    except Exception:
        return None
