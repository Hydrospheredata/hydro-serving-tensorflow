#!/bin/sh

SERVICE_ID=$1

chmod +x /app/src/main.py
sync 

cd /app
exec python3 src/main.py
