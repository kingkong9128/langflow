#!/bin/bash
set -e

export BACKEND_URL="http://localhost:8080"
export FRONTEND_PORT=7860
export LANGFLOW_MAX_FILE_SIZE_UPLOAD="${LANGFLOW_MAX_FILE_SIZE_UPLOAD:-1}"

mkdir -p /tmp/nginx
envsubst '${BACKEND_URL} ${FRONTEND_PORT} ${LANGFLOW_MAX_FILE_SIZE_UPLOAD}' \
    < /etc/nginx/conf.d/default.conf.template \
    > /tmp/nginx/default.conf

python -m langflow run &
PYTHON_PID=$!

sleep 3

if ! kill -0 $PYTHON_PID 2>/dev/null; then
    echo "ERROR: Python process failed to start"
    exit 1
fi

echo "Python started with PID $PYTHON_PID, starting nginx..."
exec /usr/sbin/nginx -g "daemon off;" -c /tmp/nginx/default.conf
