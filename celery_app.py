# celery_app.py - Background task processing with Celery
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import asyncio

try:
    from celery import Celery
    from celery.schedules import crontab
    from celery.signals import worker_ready, worker_shutdown
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None

import pandas as pd
from config import get_config
from database import get_db_manager
from services.data_service import get_data_service
from services.map_service import get_map_service, HeatmapRequest, HeatmapType, AggregationMethod
from services.cache_service import get_cache_service

logger = logging.getLogger(__name__)

# Initialize Celery app if available
if CELERY_AVAILABLE:
    config = get_config()
    
    # Celery configuration
    celery_app = Celery(
        'tapsi_food_map',
        broker=f'redis://{config.cache.host}:{config.cache.port}/1',  # Use different Redis DB
        backend=f'redis://{config.cache.host}:{config.cache.port}/2'
    )
    
    # Celery settings
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone=config.scheduler.timezone,
        enable_utc=True,
        
        # Task routing
        task_routes={
            'tapsi_food_map.fetch_data': {'queue': 'data_fetch'},
            'tapsi_food_map.generate_heatmap': {'queue': 'heatmap'},
            'tapsi_food_map.cache_warmup': {'queue': 'cache'},
            'tapsi_food_map.cleanup': {'queue': 'maintenance'}
        },
        
        # Worker settings
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        worker_max_tasks_per_child=100,
        
        # Task time limits
        task_time_limit=300,  # 5 minutes hard limit
        task_soft_time_limit=240,  # 4 minutes soft limit
        
        # Retry settings
        task_reject_on_worker_lost=True,
        task_default_retry_delay=60,
        task_max_retries=3,
        
        # Beat schedule for periodic tasks
        beat_schedule={
            'fetch-orders-daily': {
                'task': 'tapsi_food_map.fetch_orders_data',
                'schedule': crontab(hour=9, minute=0),  # Daily at 9 AM
            },
            'fetch-vendors-periodic': {
                'task': 'tapsi_food_map.fetch_vendors_data',
                'schedule': 600.0,  # Every 10 minutes
            },
            'warm-cache-hourly': {
                'task': 'tapsi_food_map.warm_cache',
                'schedule': crontab(minute=0),  # Every hour
            },
            'cleanup-cache-daily': {
                'task': 'tapsi_food_map.cleanup_old_cache',
                'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
            },
            'generate-popular-heatmaps': {
                'task': 'tapsi_food_map.pregenerate_heatmaps',
                'schedule': crontab(hour=8, minute=0),  # Daily at 8 AM
            }
        }
    )
else:
    celery_app = None

# Task definitions (only if Celery is available)
if CELERY_AVAILABLE and celery_app:
    
    @celery_app.task(bind=True, name='tapsi_food_map.fetch_orders_data')
    def fetch_orders_data_task(self, force_refresh: bool = False):
        """Background task to fetch orders data"""
        try:
            logger.info("Starting orders data fetch task")
            
            data_service = get_data_service()
            
            # Use asyncio to run async data service
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Fetch fresh data from Metabase
                result = loop.run_until_complete(
                    data_service.get_orders_data(use_cache=not force_refresh)
                )
                
                if result.success:
                    logger.info(f"Orders data fetch completed: {result.record_count} records")
                    return {
                        'success': True,
                        'record_count': result.record_count,
                        'source': result.source.value if result.source else 'unknown',
                        'fetch_time': result.fetch_time
                    }
                else:
                    logger.error(f"Orders data fetch failed: {result.error}")
                    self.retry(countdown=300, max_retries=3)  # Retry in 5 minutes
                    
            finally:
                loop.close()
            
        except Exception as e:
            logger.error(f"Orders data fetch task failed: {e}")
            self.retry(countdown=300, max_retries=3)
        
        return {'success': False, 'error': str(e)}
    
    @celery_app.task(bind=True, name='tapsi_food_map.fetch_vendors_data')
    def fetch_vendors_data_task(self):
        """Background task to fetch vendors data"""
        try:
            logger.info("Starting vendors data fetch task")
            
            data_service = get_data_service()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    data_service.get_vendors_data(use_cache=False)  # Always fetch fresh
                )
                
                if result.success:
                    logger.info(f"Vendors data fetch completed: {result.record_count} records")
                    return {
                        'success': True,
                        'record_count': result.record_count,
                        'source': result.source.value if result.source else 'unknown',
                        'fetch_time': result.fetch_time
                    }
                else:
                    logger.error(f"Vendors data fetch failed: {result.error}")
                    self.retry(countdown=120, max_retries=3)  # Retry in 2 minutes
                    
            finally:
                loop.close()
            
        except Exception as e:
            logger.error(f"Vendors data fetch task failed: {e}")
            self.retry(countdown=120, max_retries=3)
        
        return {'success': False, 'error': str(e)}
    
    @celery_app.task(bind=True, name='tapsi_food_map.generate_heatmap')
    def generate_heatmap_task(self, 
                             heatmap_type: str,
                             city_id: int,
                             business_line: Optional[str] = None,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             zoom_level: int = 11):
        """Background task to generate heatmap"""
        try:
            logger.info(f"Starting heatmap generation: {heatmap_type} for city {city_id}")
            
            map_service = get_map_service()
            
            # Create heatmap request
            request = HeatmapRequest(
                heatmap_type=HeatmapType(heatmap_type),
                city_id=city_id,
                business_line=business_line,
                start_date=start_date,
                end_date=end_date,
                zoom_level=zoom_level,
                aggregation_method=AggregationMethod.COUNT
            )
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    map_service.generate_heatmap(request)
                )
                
                if result.success:
                    logger.info(f"Heatmap generation completed: {len(result.points)} points")
                    return {
                        'success': True,
                        'point_count': len(result.points),
                        'generation_time': result.generation_time,
                        'cache_key': result.cache_key
                    }
                else:
                    logger.error(f"Heatmap generation failed: {result.error}")
                    self.retry(countdown=60, max_retries=2)
                    
            finally:
                loop.close()
            
        except Exception as e:
            logger.error(f"Heatmap generation task failed: {e}")
            self.retry(countdown=60, max_retries=2)
        
        return {'success': False, 'error': str(e)}
    
    @celery_app.task(name='tapsi_food_map.warm_cache')
    def warm_cache_task():
        """Background task to warm cache with common queries"""
        try:
            logger.info("Starting cache warming task")
            
            cache_service = get_cache_service()
            data_service = get_data_service()
            
            # Common warming operations
            warming_functions = [
                {
                    'function': lambda: asyncio.run(data_service.get_orders_data(city_id=2)),  # Tehran orders
                    'key': cache_service.create_cache_key('orders', {'city_id': 2}),
                    'data_type': 'orders'
                },
                {
                    'function': lambda: asyncio.run(data_service.get_vendors_data(city_id=2)),  # Tehran vendors
                    'key': cache_service.create_cache_key('vendors', {'city_id': 2}),
                    'data_type': 'vendors'
                },
                {
                    'function': lambda: asyncio.run(data_service.get_aggregated_data('daily', city_id=2)),
                    'key': cache_service.create_cache_key('aggregated', {'type': 'daily', 'city_id': 2}),
                    'data_type': 'aggregated'
                }
            ]
            
            successful = 0
            for func_config in warming_functions:
                try:
                    func_config['function']()
                    successful += 1
                except Exception as e:
                    logger.warning(f"Cache warming failed for {func_config['key']}: {e}")
            
            logger.info(f"Cache warming completed: {successful}/{len(warming_functions)} successful")
            
            return {
                'success': True,
                'operations_attempted': len(warming_functions),
                'successful_operations': successful
            }
            
        except Exception as e:
            logger.error(f"Cache warming task failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @celery_app.task(name='tapsi_food_map.pregenerate_heatmaps')
    def pregenerate_heatmaps_task():
        """Background task to pre-generate popular heatmaps"""
        try:
            logger.info("Starting heatmap pre-generation task")
            
            # Popular heatmap configurations
            popular_heatmaps = [
                # Tehran heatmaps
                {'heatmap_type': 'order_density', 'city_id': 2, 'business_line': 'food'},
                {'heatmap_type': 'order_density', 'city_id': 2, 'business_line': 'grocery'},
                {'heatmap_type': 'vendor_density', 'city_id': 2},
                {'heatmap_type': 'user_density', 'city_id': 2},
                
                # Mashhad heatmaps
                {'heatmap_type': 'order_density', 'city_id': 1, 'business_line': 'food'},
                {'heatmap_type': 'vendor_density', 'city_id': 1},
                
                # Shiraz heatmaps
                {'heatmap_type': 'order_density', 'city_id': 5, 'business_line': 'food'},
                {'heatmap_type': 'vendor_density', 'city_id': 5}
            ]
            
            successful = 0
            for heatmap_config in popular_heatmaps:
                try:
                    # Submit as background task
                    task_result = generate_heatmap_task.delay(**heatmap_config)
                    # Don't wait for result, just schedule
                    successful += 1
                except Exception as e:
                    logger.warning(f"Failed to schedule heatmap: {heatmap_config}, error: {e}")
            
            logger.info(f"Heatmap pre-generation scheduled: {successful}/{len(popular_heatmaps)}")
            
            return {
                'success': True,
                'heatmaps_scheduled': successful,
                'total_heatmaps': len(popular_heatmaps)
            }
            
        except Exception as e:
            logger.error(f"Heatmap pre-generation task failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @celery_app.task(name='tapsi_food_map.cleanup_old_cache')
    def cleanup_old_cache_task():
        """Background task to cleanup old cache entries"""
        try:
            logger.info("Starting cache cleanup task")
            
            cache_service = get_cache_service()
            
            # Cleanup patterns for old data
            cleanup_patterns = [
                'heatmap:*',  # Clean old heatmaps (they'll be regenerated on demand)
                'aggregated:*'  # Clean old aggregated data
            ]
            
            total_cleaned = 0
            for pattern in cleanup_patterns:
                try:
                    cleaned = cache_service.invalidate_pattern(pattern)
                    total_cleaned += cleaned
                    logger.info(f"Cleaned {cleaned} entries matching {pattern}")
                except Exception as e:
                    logger.warning(f"Failed to clean pattern {pattern}: {e}")
            
            logger.info(f"Cache cleanup completed: {total_cleaned} entries cleaned")
            
            return {
                'success': True,
                'entries_cleaned': total_cleaned
            }
            
        except Exception as e:
            logger.error(f"Cache cleanup task failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @celery_app.task(name='tapsi_food_map.database_maintenance')
    def database_maintenance_task():
        """Background task for database maintenance"""
        try:
            logger.info("Starting database maintenance task")
            
            db_manager = get_db_manager()
            
            if not db_manager._connected:
                logger.warning("Database not connected, skipping maintenance")
                return {'success': False, 'error': 'Database not connected'}
            
            maintenance_operations = []
            
            # Refresh materialized views
            try:
                with db_manager.engine.connect() as conn:
                    conn.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY vendor_stats")
                    conn.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY daily_order_stats")
                    conn.commit()
                maintenance_operations.append('materialized_views_refreshed')
            except Exception as e:
                logger.warning(f"Failed to refresh materialized views: {e}")
            
            # Update table statistics
            try:
                with db_manager.engine.connect() as conn:
                    conn.execute("ANALYZE orders")
                    conn.execute("ANALYZE vendors")
                    conn.commit()
                maintenance_operations.append('table_statistics_updated')
            except Exception as e:
                logger.warning(f"Failed to update table statistics: {e}")
            
            logger.info(f"Database maintenance completed: {', '.join(maintenance_operations)}")
            
            return {
                'success': True,
                'operations_completed': maintenance_operations
            }
            
        except Exception as e:
            logger.error(f"Database maintenance task failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # Task monitoring and health check
    @celery_app.task(name='tapsi_food_map.health_check')
    def health_check_task():
        """Health check task for monitoring"""
        try:
            logger.info("Running health check task")
            
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'services': {}
            }
            
            # Check data service
            try:
                data_service = get_data_service()
                stats = data_service.get_service_stats()
                health_status['services']['data_service'] = {
                    'status': 'healthy',
                    'stats': stats
                }
            except Exception as e:
                health_status['services']['data_service'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
            
            # Check cache service
            try:
                cache_service = get_cache_service()
                stats = cache_service.get_cache_statistics()
                health_status['services']['cache_service'] = {
                    'status': 'healthy',
                    'stats': stats
                }
            except Exception as e:
                health_status['services']['cache_service'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
            
            # Check database
            try:
                db_manager = get_db_manager()
                if db_manager.health_check():
                    db_stats = db_manager.get_database_stats()
                    health_status['services']['database'] = {
                        'status': 'healthy',
                        'stats': db_stats.__dict__ if hasattr(db_stats, '__dict__') else str(db_stats)
                    }
                else:
                    health_status['services']['database'] = {
                        'status': 'unhealthy',
                        'error': 'Health check failed'
                    }
            except Exception as e:
                health_status['services']['database'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
            
            # Overall health
            unhealthy_services = [
                service for service, status in health_status['services'].items()
                if status['status'] == 'unhealthy'
            ]
            
            health_status['overall_status'] = 'unhealthy' if unhealthy_services else 'healthy'
            health_status['unhealthy_services'] = unhealthy_services
            
            logger.info(f"Health check completed: {health_status['overall_status']}")
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check task failed: {e}")
            return {
                'overall_status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Signal handlers
    @worker_ready.connect
    def worker_ready_handler(sender=None, **kwargs):
        """Handler for when worker is ready"""
        logger.info("Celery worker is ready")
        
        # Schedule initial cache warming
        warm_cache_task.delay()
    
    @worker_shutdown.connect
    def worker_shutdown_handler(sender=None, **kwargs):
        """Handler for when worker is shutting down"""
        logger.info("Celery worker is shutting down")

else:
    # Fallback implementations when Celery is not available
    logger.warning("Celery not available, background tasks will be disabled")
    
    def fetch_orders_data_task(*args, **kwargs):
        logger.warning("Celery not available, orders fetch task skipped")
        return {'success': False, 'error': 'Celery not available'}
    
    def fetch_vendors_data_task(*args, **kwargs):
        logger.warning("Celery not available, vendors fetch task skipped")
        return {'success': False, 'error': 'Celery not available'}
    
    def generate_heatmap_task(*args, **kwargs):
        logger.warning("Celery not available, heatmap generation task skipped")
        return {'success': False, 'error': 'Celery not available'}

# Helper functions
def get_celery_app():
    """Get Celery application instance"""
    return celery_app

def is_celery_available() -> bool:
    """Check if Celery is available"""
    return CELERY_AVAILABLE and celery_app is not None

def submit_background_task(task_name: str, *args, **kwargs):
    """Submit background task if Celery is available"""
    if not is_celery_available():
        logger.warning(f"Cannot submit task {task_name}: Celery not available")
        return None
    
    try:
        task = celery_app.send_task(task_name, args=args, kwargs=kwargs)
        logger.info(f"Submitted background task: {task_name} (ID: {task.id})")
        return task
    except Exception as e:
        logger.error(f"Failed to submit background task {task_name}: {e}")
        return None

def get_task_status(task_id: str):
    """Get status of background task"""
    if not is_celery_available():
        return {'status': 'error', 'error': 'Celery not available'}
    
    try:
        task_result = celery_app.AsyncResult(task_id)
        return {
            'status': task_result.status,
            'result': task_result.result,
            'traceback': task_result.traceback
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# Startup function for integration with main app
def initialize_celery():
    """Initialize Celery integration"""
    if is_celery_available():
        logger.info("Celery integration initialized")
        return True
    else:
        logger.warning("Celery not available, background processing disabled")
        return False