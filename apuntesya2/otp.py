import os, hmac, hashlib, random
from datetime import datetime, timedelta
from sqlalchemy import text

PEPPER = os.getenv("OTP_PEPPER", "pepper-super-secreto")

def _hash(code: str) -> str:
    return hmac.new(PEPPER.encode(), code.encode(), hashlib.sha256).hexdigest()

def generate_code() -> str:
    return f"{random.randint(0, 999999):06d}"

def create_otp(Session, user_id: int, channel: str = "sms", ttl_minutes=10):
    code = generate_code()
    code_hash = _hash(code)
    expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
    with Session() as s:
        s.execute(text(
            "INSERT INTO otps (user_id, channel, code_hash, expires_at) VALUES (:u,:c,:h,:e)"
        ), {"u": user_id, "c": channel, "h": code_hash, "e": expires_at})
        s.commit()
    return code

def verify_otp(Session, user_id: int, channel: str, code: str, max_attempts=5) -> bool:
    code_hash = _hash(code)
    with Session() as s:
        row = s.execute(text(
            "SELECT id, code_hash, expires_at, attempts FROM otps WHERE user_id=:u AND channel=:c "
            "ORDER BY id DESC LIMIT 1"
        ), {"u": user_id, "c": channel}).first()
        if not row: return False
        oid, chash, exp, attempts = row
        if attempts >= max_attempts: return False
        exp_dt = exp if hasattr(exp, 'year') else datetime.fromisoformat(str(exp))
        if exp_dt < datetime.utcnow(): return False
        ok = hmac.compare_digest(chash, code_hash)
        s.execute(text("UPDATE otps SET attempts=attempts+1 WHERE id=:id"), {"id": oid})
        s.commit()
        return ok
