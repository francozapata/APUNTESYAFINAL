#!/usr/bin/env bash
set -e
export PORT=${PORT:-10000}
export WORKERS=${WORKERS:-2}
export THREADS=${THREADS:-2}
export TIMEOUT=${TIMEOUT:-120}
export APP_MODULE=${APP_MODULE:-apuntesya2.app:app}
echo "[start.sh] Gunicorn -> $APP_MODULE on :$PORT (w=$WORKERS t=$THREADS timeout=$TIMEOUT)"
exec gunicorn "$APP_MODULE" \
  --workers "$WORKERS" \
  --threads "$THREADS" \
  --timeout "$TIMEOUT" \
  --bind "0.0.0.0:${PORT}"
