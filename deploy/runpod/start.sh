#!/usr/bin/env bash
# Prepares the data directories and the database on first boot, then runs supervisord.
set -euo pipefail

data="${SPARKY_DATA_DIR:-/workspace}"
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
