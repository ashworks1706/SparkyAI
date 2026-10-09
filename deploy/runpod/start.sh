#!/usr/bin/env bash
# Prepares the data directories and the database on first boot, then runs supervisord.
# With SPARKY_PLATFORM__ENABLED=true the stores live on Platform, so the datastores, scraper
# and search are not started and no database is created. With SPARKY_POD_ROLE=models only the two model
# servers run, for an engine and bot that run elsewhere.
set -euo pipefail

data="${SPARKY_DATA_DIR:-/workspace}"

if [ -n "${SPARKY_MODELS_API_KEY:-}" ]; then
    export SPARKY_MODELS_HOST=0.0.0.0
    export SPARKY_MODEL__API_KEY="$SPARKY_MODELS_API_KEY" SPARKY_SUMMARY__API_KEY="$SPARKY_MODELS_API_KEY" \
        SPARKY_EMBEDDING__API_KEY="$SPARKY_MODELS_API_KEY"
else
    export SPARKY_MODELS_HOST=127.0.0.1
fi

export SPARKY_RUN_APPS=true
if [ "${SPARKY_POD_ROLE:-all}" = "models" ]; then
    if [ -z "${SPARKY_MODELS_API_KEY:-}" ]; then
        echo "SPARKY_POD_ROLE=models needs SPARKY_MODELS_API_KEY" >&2
        exit 1
    fi
    export SPARKY_RUN_APPS=false SPARKY_STANDALONE_SERVICES=false
    mkdir -p "$data/models"
    exec /usr/bin/supervisord -c /etc/sparky/supervisord.conf
fi

if [ "${SPARKY_PLATFORM__ENABLED:-false}" = "true" ]; then
    export SPARKY_STANDALONE_SERVICES=false
    mkdir -p "$data/models"
    exec /usr/bin/supervisord -c /etc/sparky/supervisord.conf
fi
export SPARKY_STANDALONE_SERVICES=true

pgbin=/usr/lib/postgresql/16/bin
mkdir -p "$data"/{postgres,redis,minio,models,auth,backups}
chown postgres:postgres "$data/postgres" "$data/backups"
chmod 700 "$data/postgres"

if [ ! -s "$data/postgres/PG_VERSION" ]; then
    su postgres -c "$pgbin/initdb -D $data/postgres -U postgres --auth-local=trust --auth-host=scram-sha-256"
    su postgres -c "$pgbin/pg_ctl -D $data/postgres -o '-c listen_addresses=' -w start"
    su postgres -c "psql -v ON_ERROR_STOP=1 -d postgres" <<'SQL'
create role sparky login password 'sparky';
create database sparky owner sparky;
\c sparky
create extension if not exists vector;
SQL
    su postgres -c "$pgbin/pg_ctl -D $data/postgres -w stop"
fi

exec /usr/bin/supervisord -c /etc/sparky/supervisord.conf
