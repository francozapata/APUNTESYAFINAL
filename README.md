# ApuntesYa2

Plataforma en Flask para compartir y vender apuntes (PDF). Incluye:
- Registro/login con filtros por Universidad/Facultad/Carrera.
- Subida de PDFs gratuitos o pagos.
- Búsqueda y filtrado por Universidad/Facultad/Carrera.
- Vinculación con Mercado Pago vía OAuth para que cada vendedor cobre directo.
- Cobro con **Checkout Pro** creando `preferences` con el **token del vendedor** y `marketplace_fee`
  para retener la **comisión del 5%** a la cuenta de la plataforma.
- Webhook para actualizar compras y habilitar descargas.

## Requisitos
- Python 3.10+
- Pip
- Cuenta de Mercado Pago (plataforma) y una app con OAuth habilitado.
- Variables en `.env` (copiar desde `.env.example`).

## Setup rápido
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copiá .env.example a .env y completa/valida valores
cp .env.example .env
# Windows (PowerShell): Copy-Item .env.example .env
# Asegurate de estar parado dentro de la carpeta del proyecto (donde está app.py).

# Inicializar DB
python create_db.py

# Correr
flask --app app run -h 0.0.0.0 -p 5000 --debug
```
Abrí http://localhost:5000

## Flujo de cobro (Marketplace)
1. El vendedor va a **Perfil → Conectar con Mercado Pago**. Autoriza la app (OAuth).
2. Guardamos su `access_token` (del vendedor) cifrado y su `user_id` de MP.
3. Al comprar un apunte pago:
   - Creamos la **preference** con el **token del vendedor** (así el dinero va a su cuenta).
   - Calculamos `marketplace_fee = price * MP_PLATFORM_FEE_PERCENT / 100` (monto fijo).
   - Usamos `notification_url` → `/mp/webhook` para recibir el estado del pago.
4. Cuando MP envía `payment.approved`, habilitamos la **descarga** al comprador.

> Nota: Para testear con sandbox, activá modo test en tu cuenta y usa usuarios de prueba.

## Consideraciones
- Subimos **solo PDF**. Tamaño máximo configurable.
- Los archivos se guardan en `uploads/` y se referencian en DB.
- Descarga:
  - Gratuitos: logueado.
  - Pagos: sólo si sos el vendedor o si la compra está **aprobada**.
- Seguridad básica (login, ownership, verificación de tipos). Recomendado poner Nginx, HTTPS, etc.
