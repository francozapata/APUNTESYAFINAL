import smtplib, ssl, base64
from email.message import EmailMessage
from flask import current_app, url_for, render_template

def _bool(val, default=False):
    if val is None:
        return default
    return str(val).lower() in ("1","true","yes","on")

def send_reset_email(to_email: str, token: str) -> bool:
    reset_url = url_for('auth_reset.reset_password', token=token, _external=True)
    print(f"[ApuntesYa] RESET LINK for {to_email}: {reset_url}")

    # Prepare logo (base64) for HTML template
    logo_b64 = current_app.config.get('EMAIL_LOGO_BASE64')
    if not logo_b64:
        try:
            from pathlib import Path
            logo_file = Path(current_app.root_path) / 'static' / 'img' / 'logo.png'
            if logo_file.exists():
                data = logo_file.read_bytes()
                logo_b64 = 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')
            else:
                logo_b64 = ''
        except Exception:
            logo_b64 = ''

    html = render_template('emails/reset_password.html', reset_url=reset_url, logo_b64=logo_b64)
    plain = f"""Hola,

Recibimos una solicitud para restablecer la contraseña de tu cuenta.

Ingresá al siguiente enlace para crear una nueva contraseña (expira en 1 hora):
{reset_url}

Si no solicitaste este cambio, podés ignorar este mail.

Equipo ApuntesYa
"""

    if not _bool(current_app.config.get('ENABLE_SMTP')):
        print("[ApuntesYa] SMTP desactivado (ENABLE_SMTP!=true). No se envió correo.")
        return True

    msg = EmailMessage()
    msg['Subject'] = 'Restablecé tu contraseña - ApuntesYa'
    sender = current_app.config.get('MAIL_DEFAULT_SENDER', 'no-reply@localhost')
    msg['From'] = sender
    msg['To'] = to_email
    msg.set_content(plain)
    msg.add_alternative(html, subtype='html')

    server = current_app.config.get('MAIL_SERVER')
    port = int(current_app.config.get('MAIL_PORT', 587))
    username = current_app.config.get('MAIL_USERNAME')
    password = current_app.config.get('MAIL_PASSWORD')
    use_tls = _bool(current_app.config.get('MAIL_USE_TLS'), True)
    use_ssl = _bool(current_app.config.get('MAIL_USE_SSL'), False)
    timeout = int(current_app.config.get('MAIL_TIMEOUT', 20))

    if not server or not username or not password:
        print('[ApuntesYa] SMTP no configurado (faltan MAIL_SERVER/MAIL_USERNAME/MAIL_PASSWORD).')
        return False

    try:
        if use_ssl or port == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(server, port or 465, context=context, timeout=timeout) as smtp:
                smtp.login(username, password)
                smtp.send_message(msg)
        else:
            context = ssl.create_default_context()
            with smtplib.SMTP(server, port or 587, timeout=timeout) as smtp:
                if use_tls:
                    smtp.starttls(context=context)
                smtp.login(username, password)
                smtp.send_message(msg)
        print(f"[ApuntesYa] Email enviado correctamente a {to_email}")
        return True
    except Exception as e:
        print(f"[ApuntesYa] ERROR enviando email a {to_email}: {e}")
        try:
            print("[ApuntesYa] Reintentando con SSL 465...")
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(server, 465, context=context, timeout=timeout) as smtp:
                smtp.login(username, password)
                smtp.send_message(msg)
            print(f"[ApuntesYa] Email enviado correctamente a {to_email} por SSL 465 (fallback)")
            return True
        except Exception as e2:
            print(f"[ApuntesYa] Segundo intento falló: {e2}")
            return False
