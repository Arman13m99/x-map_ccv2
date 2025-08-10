# routes/health.py - Health check and monitoring routes
import logging
from datetime import datetime
from flask import Blueprint, jsonify, request

from config import get_config
from database import get_db_manager
from services.data_service import get_data_service
from services.cache_service import get_cache_service
from scheduler import get_scheduler
from celery_app import get_celery_app, is_celery_available

logger = logging.getLogger(__name__)

# Create blueprint
health_bp = Blueprint('health', __name__, url_prefix='/health')

@health_bp.route('/', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    try:
        config = get_config()
        data_service = get_data_service()
        cache_service = get_cache_service()
        db_manager = get_db_manager()
        scheduler = get_scheduler()
        
        # Collect health status from all components
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0-production-phase2',
            'components': {}
        }
        
        # Check data service
        try:
            service_stats = data_service.get_service_stats()
            health_status['components']['data_service'] = {
                'status': 'healthy',
                'stats': service_stats
            }
        except Exception as e:
            health_status['components']['data_service'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Check cache service
        try:
            cache_stats = cache_service.get_cache_statistics()
            cache_healthy = cache_service.cache_manager.health_check()
            health_status['components']['cache'] = {
                'status': 'healthy' if cache_healthy else 'degraded',
                'stats': cache_stats
            }
        except Exception as e:
            health_status['components']['cache'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Check database
        try:
            if db_manager._connected:
                db_healthy = db_manager.health_check()
                db_stats = db_manager.get_database_stats()
                health_status['components']['database'] = {
                    'status': 'healthy' if db_healthy else 'unhealthy',
                    'stats': db_stats.__dict__ if hasattr(db_stats, '__dict__') else str(db_stats)
                }
            else:
                health_status['components']['database'] = {
                    'status': 'not_configured',
                    'message': 'Database not configured'
                }
        except Exception as e:
            health_status['components']['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Check background scheduler
        try:
            scheduler_status = scheduler.get_data_status()
            health_status['components']['scheduler'] = {
                'status': 'healthy' if scheduler.scheduler and scheduler.scheduler.running else 'unhealthy',
                'stats': scheduler_status
            }
        except Exception as e:
            health_status['components']['scheduler'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Check Celery (if available)
        try:
            if is_celery_available():
                celery_app = get_celery_app()
                # Simple check - try to inspect active tasks
                active_tasks = celery_app.control.inspect().active()
                health_status['components']['celery'] = {
                    'status': 'healthy',
                    'active_tasks': len(active_tasks) if active_tasks else 0
                }
            else:
                health_status['components']['celery'] = {
                    'status': 'not_available',
                    'message': 'Celery not configured'
                }
        except Exception as e:
            health_status['components']['celery'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Determine overall health status
        unhealthy_components = [
            name for name, status in health_status['components'].items()
            if status['status'] == 'unhealthy'
        ]
        
        if unhealthy_components:
            health_status['status'] = 'unhealthy'
            health_status['unhealthy_components'] = unhealthy_components
        
        # Return appropriate HTTP status
        status_code = 503 if health_status['status'] == 'unhealthy' else 200
        
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@health_bp.route('/ready', methods=['GET'])
def readiness_check():
    """Kubernetes readiness probe endpoint"""
    try:
        scheduler = get_scheduler()
        data_status = scheduler.get_data_status()
        
        # Application is ready if it has some data loaded
        ready = (
            data_status['orders']['loaded'] or 
            data_status['vendors']['loaded']
        )
        
        if ready:
            return jsonify({
                'status': 'ready',
                'timestamp': datetime.now().isoformat(),
                'data_status': data_status
            }), 200
        else:
            return jsonify({
                'status': 'not_ready',
                'message': 'Data is still loading',
                'timestamp': datetime.now().isoformat(),
                'data_status': data_status
            }), 503
            
    except Exception as e:
        return jsonify({
            'status': 'not_ready',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@health_bp.route('/live', methods=['GET'])
def liveness_check():
    """Kubernetes liveness probe endpoint"""
    return jsonify({
        'status': 'alive',
        'timestamp': datetime.now().isoformat()
    }), 200

@health_bp.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus-style metrics endpoint"""
    try:
        config = get_config()
        data_service = get_data_service()
        cache_service = get_cache_service()
        
        # Collect metrics
        service_stats = data_service.get_service_stats()
        cache_stats = cache_service.get_cache_statistics()
        
        # Format as simple key-value pairs (Prometheus style)
        metrics_text = []
        
        # Service metrics
        metrics_text.append(f"# HELP data_service_total_calls Total data service calls")
        metrics_text.append(f"# TYPE data_service_total_calls counter")
        metrics_text.append(f"data_service_total_calls {service_stats.get('total_calls', 0)}")
        
        metrics_text.append(f"# HELP data_service_cache_hit_rate Cache hit rate percentage")
        metrics_text.append(f"# TYPE data_service_cache_hit_rate gauge")
        metrics_text.append(f"data_service_cache_hit_rate {service_stats.get('cache_hit_rate', 0)}")
        
        # Cache metrics
        global_metrics = cache_stats.get('global_metrics', {})
        metrics_text.append(f"# HELP cache_total_requests Total cache requests")
        metrics_text.append(f"# TYPE cache_total_requests counter")
        metrics_text.append(f"cache_total_requests {global_metrics.get('total_requests', 0)}")
        
        metrics_text.append(f"# HELP cache_hit_rate Global cache hit rate")
        metrics_text.append(f"# TYPE cache_hit_rate gauge")
        metrics_text.append(f"cache_hit_rate {global_metrics.get('global_hit_rate', 0)}")
        
        return '\n'.join(metrics_text), 200, {'Content-Type': 'text/plain'}
        
    except Exception as e:
        return f"# Error collecting metrics: {str(e)}", 500, {'Content-Type': 'text/plain'}

@health_bp.route('/stats', methods=['GET'])
def detailed_stats():
    """Detailed statistics endpoint for monitoring"""
    try:
        data_service = get_data_service()
        cache_service = get_cache_service()
        db_manager = get_db_manager()
        scheduler = get_scheduler()
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'data_service': data_service.get_service_stats(),
            'cache_service': cache_service.get_cache_statistics(),
            'scheduler': scheduler.get_data_status()
        }
        
        # Add database stats if available
        if db_manager._connected:
            try:
                db_stats = db_manager.get_database_stats()
                stats['database'] = db_stats.__dict__ if hasattr(db_stats, '__dict__') else str(db_stats)
            except Exception as e:
                stats['database'] = {'error': str(e)}
        
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500