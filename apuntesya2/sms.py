import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from base64 import b64encode

def send_sms(to_e164: str, body: str) -> bool:
    sid   = os.getenv("TWILIO_ACCOUNT_SID","")
    token = os.getenv("TWILIO_AUTH_TOKEN","")
    from_ = os.getenv("TWILIO_FROM_SMS","")
    if not (sid and token and from_):
        print("[OTP] Falta config Twilio SMS"); return False
    data = urlencode({"To": to_e164, "From": from_, "Body": body}).encode()
    req = Request(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json", data=data)
    auth = b64encode(f"{sid}:{token}".encode()).decode()
    req.add_header("Authorization", f"Basic {auth}")
    try:
        with urlopen(req, timeout=15) as r:
            print("[OTP] Twilio SMS status:", r.status)
            return r.status in (200, 201)
    except Exception as e:
        print("[OTP] Twilio SMS error:", e); return False
