#!/usr/bin/env bash
# Dumps the database once a day into the backups directory and keeps the newest SPARKY_BACKUP_KEEP.
set -euo pipefail

dir="${SPARKY_DATA_DIR:-/workspace}/backups"
keep="${SPARKY_BACKUP_KEEP:-7}"
sleep 600
while true; do
    file="$dir/sparky-$(date -u +%Y%m%dT%H%M%SZ).dump"
    if pg_dump --format=custom --file="$file.tmp" "postgres://sparky:sparky@127.0.0.1:5432/sparky"; then
        mv "$file.tmp" "$file"
        echo "backup: wrote $file"
    else
        rm -f "$file.tmp"
        echo "backup: pg_dump failed"
    fi
    ls -1t "$dir"/sparky-*.dump 2>/dev/null | tail -n +"$((keep + 1))" | xargs -r rm -f
    sleep 86400
done
