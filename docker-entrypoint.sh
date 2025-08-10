#!/bin/bash
# docker-entrypoint.sh - Production container entrypoint script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Set default values
export WORKERS=${WORKERS:-4}
export TIMEOUT=${TIMEOUT:-120}
export KEEPALIVE=${KEEPALIVE:-5}
export MAX_REQUESTS=${MAX_REQUESTS:-1000}
export MAX_REQUESTS_JITTER=${MAX_REQUESTS_JITTER:-100}
export PORT=${PORT:-5000}
export HOST=${HOST:-0.0.0.0}

log_info "Starting Tapsi Food Map Dashboard v${VERSION:-2.0.0}"
log_info "Container: $(hostname)"
log_info "User: $(whoami)"
log_info "Working Directory: $(pwd)"

# Wait for dependencies
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}
    
    log_info "Waiting for $service_name ($host:$port) to be ready..."
    
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port" > /dev/null 2>&1; then
            log_info "$service_name is ready!"
            return 0
        fi
        sleep 1
    done
    
    log_error "$service_name ($host:$port) is not available after ${timeout}s"
    return 1
}

# Wait for Redis if configured
if [ -n "$REDIS_HOST" ] && [ -n "$REDIS_PORT" ]; then
    wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis" 30
fi

# Wait for PostgreSQL if configured
if [ -n "$DB_HOST" ] && [ -n "$DB_PORT" ]; then
    wait_for_service "$DB_HOST" "$DB_PORT" "PostgreSQL" 60
elif [ -n "$DATABASE_URL" ]; then
    # Extract host and port from DATABASE_URL
    if [[ $DATABASE_URL =~ postgresql://[^@]+@([^:/]+):([0-9]+)/ ]]; then
        DB_HOST="${BASH_REMATCH[1]}"
        DB_PORT="${BASH_REMATCH[2]}"
        wait_for_service "$DB_HOST" "$DB_PORT" "PostgreSQL" 60
    fi
fi

# Run database migrations if needed
run_migrations() {
    log_info "Running database migrations..."
    python -c "
from database import get_db_manager, get_db_migrator
db_manager = get_db_manager()
if db_manager._connected:
    migrator = get_db_migrator()
    success = migrator.migrate_to_latest()
    if success:
        print('Migrations completed successfully')
    else:
        print('Migration failed')
        exit(1)
else:
    print('Database not connected, skipping migrations')
"
}

# Validate configuration
validate_config() {
    log_info "Validating configuration..."
    python -c "
from config import get_config
config = get_config()
if config.validate():
    print('Configuration is valid')
else:
    print('Configuration validation failed')
    exit(1)
"
}

# Health check
health_check() {
    log_info "Performing health check..."
    python -c "
import requests
import sys
try:
    response = requests.get(f'http://localhost:${PORT}/health', timeout=10)
    if response.status_code == 200:
        print('Health check passed')
        sys.exit(0)
    else:
        print(f'Health check failed: HTTP {response.status_code}')
        sys.exit(1)
except Exception as e:
    print(f'Health check failed: {e}')
    sys.exit(1)
"
}

# Main execution logic
case "$1" in
    "gunicorn")
        log_info "Starting Gunicorn WSGI server..."
        
        validate_config
        
        # Run migrations if database is available
        if [ "$RUN_MIGRATIONS" = "true" ]; then
            run_migrations
        fi
        
        # Create Gunicorn configuration
        cat > /tmp/gunicorn.conf.py << EOF
# Gunicorn production configuration
bind = "${HOST}:${PORT}"
workers = ${WORKERS}
worker_class = "gevent"
worker_connections = 1000
max_requests = ${MAX_REQUESTS}
max_requests_jitter = ${MAX_REQUESTS_JITTER}
timeout = ${TIMEOUT}
keepalive = ${KEEPALIVE}
preload_app = True
daemon = False
user = $(id -u)
group = $(id -g)
tmp_upload_dir = None
secure_scheme_headers = {
    'X-FORWARDED-PROTOCOL': 'ssl',
    'X-FORWARDED-PROTO': 'https',
    'X-FORWARDED-SSL': 'on'
}
forwarded_allow_ips = '*'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
accesslog = '-'
errorlog = '-'
loglevel = '${LOG_LEVEL:-info}'
capture_output = True
enable_stdio_inheritance = True

def when_ready(server):
    server.log.info("Tapsi Food Map Dashboard is ready to serve requests")

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)
EOF
        
        log_info "Gunicorn configuration:"
        log_info "  Workers: $WORKERS"
        log_info "  Bind: $HOST:$PORT"
        log_info "  Timeout: $TIMEOUT seconds"
        log_info "  Max Requests: $MAX_REQUESTS"
        
        exec gunicorn app:app -c /tmp/gunicorn.conf.py
        ;;
    
    "celery-worker")
        log_info "Starting Celery worker..."
        
        validate_config
        
        # Celery worker configuration
        export CELERY_WORKER_CONCURRENCY=${CELERY_WORKER_CONCURRENCY:-4}
        export CELERY_WORKER_PREFETCH=${CELERY_WORKER_PREFETCH:-1}
        export CELERY_WORKER_MAX_TASKS=${CELERY_WORKER_MAX_TASKS:-100}
        
        log_info "Celery worker configuration:"
        log_info "  Concurrency: $CELERY_WORKER_CONCURRENCY"
        log_info "  Prefetch: $CELERY_WORKER_PREFETCH"
        log_info "  Max Tasks per Child: $CELERY_WORKER_MAX_TASKS"
        
        exec celery -A celery_app worker \
            --loglevel=${LOG_LEVEL:-info} \
            --concurrency=$CELERY_WORKER_CONCURRENCY \
            --prefetch-multiplier=$CELERY_WORKER_PREFETCH \
            --max-tasks-per-child=$CELERY_WORKER_MAX_TASKS \
            --without-heartbeat \
            --without-mingle \
            --without-gossip
        ;;
    
    "celery-beat")
        log_info "Starting Celery beat scheduler..."
        
        validate_config
        
        exec celery -A celery_app beat \
            --loglevel=${LOG_LEVEL:-info} \
            --schedule=/tmp/celerybeat-schedule \
            --pidfile=/tmp/celerybeat.pid
        ;;
    
    "celery-flower")
        log_info "Starting Celery Flower monitoring..."
        
        export FLOWER_PORT=${FLOWER_PORT:-5555}
        export FLOWER_BASIC_AUTH=${FLOWER_BASIC_AUTH:-admin:admin}
        
        exec celery -A celery_app flower \
            --port=$FLOWER_PORT \
            --basic_auth=$FLOWER_BASIC_AUTH
        ;;
    
    "migrate")
        log_info "Running database migrations only..."
        validate_config
        run_migrations
        log_info "Migration completed"
        ;;
    
    "health-check")
        health_check
        ;;
    
    "shell")
        log_info "Starting Python shell..."
        exec python -i -c "
from config import get_config
from database import get_db_manager
from services.data_service import get_data_service
from services.cache_service import get_cache_service
print('Tapsi Food Map Dashboard - Interactive Shell')
print('Available objects: config, db_manager, data_service, cache_service')
config = get_config()
db_manager = get_db_manager()
data_service = get_data_service()
cache_service = get_cache_service()
"
        ;;
    
    "bash")
        log_info "Starting bash shell..."
        exec /bin/bash
        ;;
    
    *)
        log_info "Available commands:"
        log_info "  gunicorn      - Start WSGI server (default)"
        log_info "  celery-worker - Start Celery worker"
        log_info "  celery-beat   - Start Celery scheduler"
        log_info "  celery-flower - Start Celery monitoring"
        log_info "  migrate       - Run database migrations"
        log_info "  health-check  - Run health check"
        log_info "  shell         - Start Python shell"
        log_info "  bash          - Start bash shell"
        
        if [ -n "$1" ]; then
            log_error "Unknown command: $1"
            exit 1
        else
            log_info "No command specified, starting Gunicorn..."
            exec "$0" gunicorn
        fi
        ;;
esac