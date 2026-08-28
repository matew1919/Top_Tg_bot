#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ENV_FILE=${VACANCY_ENV_FILE:-$SCRIPT_DIR/.env}

exec "$SCRIPT_DIR/.venv/bin/python" \
  "$SCRIPT_DIR/vacancy_watcher.py" \
  --delay-first \
  --env-file "$ENV_FILE"
