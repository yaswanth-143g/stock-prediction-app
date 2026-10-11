#!/bin/bash
set -e

echo "🚀 Starting Enterprise NSE Stock Prediction & Quantitative Portfolio Platform..."

# Wait for MySQL Database container readiness if configured
if [ "$DB_TYPE" = "mysql" ]; then
    DB_HOST_VAL=${DB_HOST:-db}
    DB_PORT_VAL=${DB_PORT:-3306}
    echo "⌛ Waiting for MySQL Database at $DB_HOST_VAL:$DB_PORT_VAL..."
    for i in $(seq 1 30); do
        if nc -z "$DB_HOST_VAL" "$DB_PORT_VAL" 2>/dev/null; then
            echo "✅ MySQL Database connection established!"
            break
        fi
        sleep 1
    done
fi

# Pre-train or load Machine Learning Model artifacts
echo "🧠 Initializing ML Ensemble Models & Quant Engines..."
python -c "
try:
    from predict import ModelManager
    manager = ModelManager()
    print('✅ ML Model Manager loaded successfully.')
except Exception as e:
    print(f'⚠️ Warning during ML preloading: {e}')
"


# Launch Gunicorn WSGI Server (Port 5000)
WORKER_CLASS=${GUNICORN_WORKER_CLASS:-gthread}
WORKERS=${GUNICORN_WORKERS:-1}
THREADS=${GUNICORN_THREADS:-8}

echo "⚡ Launching Gunicorn WSGI Server ($WORKERS Workers, $THREADS Threads, $WORKER_CLASS Engine on port ${PORT:-5000})..."
exec gunicorn --worker-class "$WORKER_CLASS" -w "$WORKERS" --threads "$THREADS" --bind "0.0.0.0:${PORT:-5000}" app:app

