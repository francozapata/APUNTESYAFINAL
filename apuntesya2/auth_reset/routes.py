from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app
from werkzeug.security import generate_password_hash
from .tokens import generate_token, confirm_token
from ..models import User
from .email_utils import send_reset_email

bp = Blueprint('auth_reset', __name__, template_folder='../templates')

def _session():
    from ..app import Session
    return Session

@bp.route('/reset_password_request', methods=['GET','POST'])
def reset_password_request():
    if request.method == 'POST':
        email = request.form.get('email','').strip().lower()
        Session = _session()
        with Session() as s:
            user = s.query(User).filter(User.email == email).first()
            if user:
                sent_ok = send_reset_email(user.email, generate_token(user.email))
                if not sent_ok:
                    print('[ApuntesYa] Aviso: no se pudo enviar el correo (SMTP). El enlace se imprimió en consola.')
        flash('Si existe una cuenta con ese mail, te enviamos instrucciones por email.', 'info')
        return redirect(url_for('login')) if 'login' in current_app.view_functions else redirect(url_for('index'))
    return render_template('auth_reset/reset_password_request.html')

@bp.route('/reset_password/<token>', methods=['GET','POST'])
def reset_password(token):
    email = confirm_token(token, int(current_app.config.get('PASSWORD_RESET_EXPIRATION', 3600)))
    if not email:
        flash('El enlace es inválido o expiró.', 'warning')
        return redirect(url_for('auth_reset.reset_password_request'))

    Session = _session()
    with Session() as s:
        user = s.query(User).filter(User.email == email).first()
        if not user:
            flash('El enlace es inválido o expiró.', 'warning')
            return redirect(url_for('auth_reset.reset_password_request'))

        if request.method == 'POST':
            password = request.form.get('password','')
            password2 = request.form.get('password2','')
            if len(password) < 8:
                flash('La contraseña debe tener al menos 8 caracteres.', 'danger')
                return render_template('auth_reset/reset_password.html', token=token)
            if password != password2:
                flash('Las contraseñas no coinciden.', 'danger')
                return render_template('auth_reset/reset_password.html', token=token)
            user.password_hash = generate_password_hash(password)
            s.commit()
            flash('Tu contraseña fue actualizada. Podés iniciar sesión.', 'success')
            return redirect(url_for('login')) if 'login' in current_app.view_functions else redirect(url_for('index'))
    return render_template('auth_reset/reset_password.html', token=token)
