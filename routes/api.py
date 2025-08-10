# routes/api.py - Main API routes
import logging
import time
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from flask import Blueprint, jsonify, request, g

from config import get_config
from services.data_service import get_data_service
from services.map_service import get_map_service, HeatmapRequest, HeatmapType, AggregationMethod
from services.cache_service import get_cache_service
from scheduler import get_scheduler
from celery_app import submit_background_task, get_task_status

logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Middleware for request timing
@api_bp.before_request
def before_request():
    """Record request start time"""
    g.start_time = time.time()

@api_bp.after_request
def after_request(response):
    """Add timing headers and log requests"""
    if hasattr(g, 'start_time'):
        request_time = time.time() - g.start_time
        response.headers['X-Response-Time'] = str(round(request_time * 1000, 2))
        
        # Log slow requests
        if request_time > 2.0:  # Log requests taking > 2 seconds
            logger.warning(f"Slow request: {request.method} {request.path} took {request_time:.2f}s")
    
    return response

# Error handlers
@api_bp.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    return jsonify({
        'error': 'Bad Request',
        'message': str(error.description),
        'status_code': 400
    }), 400

@api_bp.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found',
        'status_code': 404
    }), 404

@api_bp.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred',
        'status_code': 500
    }), 500

@api_bp.route('/initial-data', methods=['GET'])
def get_initial_data():
    """Get initial dashboard configuration data"""
    try:
        config = get_config()
        scheduler = get_scheduler()
        
        # Get fresh data status
        data_status = scheduler.get_data_status()
        
        # If no data is loaded, return loading state
        if not (data_status['orders']['loaded'] or data_status['vendors']['loaded']):
            return jsonify({
                'loading': True,
                'message': 'Data is being loaded. Please wait a moment...',
                'data_status': data_status
            }), 202
        
        # Get basic configuration data
        cities = [{"id": cid, "name": name} for cid, name in config.app.city_id_map.items()]
        
        # Get business lines from cache or scheduler data
        business_lines = []
        try:
            orders_data = scheduler.get_orders_data()
            if orders_data is not None and not orders_data.empty and 'business_line' in orders_data.columns:
                business_lines = sorted(orders_data['business_line'].dropna().unique().tolist())
        except Exception as e:
            logger.warning(f"Failed to get business lines: {e}")
        
        # Get vendor metadata
        vendor_statuses = []
        vendor_grades = []
        try:
            vendors_data = scheduler.get_vendors_data()
            if vendors_data is not None and not vendors_data.empty:
                if 'status_id' in vendors_data.columns:
                    vendor_statuses = sorted(vendors_data['status_id'].dropna().astype(int).unique().tolist())
                if 'grade' in vendors_data.columns:
                    vendor_grades = sorted(vendors_data['grade'].dropna().unique().tolist())
        except Exception as e:
            logger.warning(f"Failed to get vendor metadata: {e}")
        
        return jsonify({
            'cities': cities,
            'business_lines': business_lines,
            'vendor_statuses': vendor_statuses,
            'vendor_grades': vendor_grades,
            'data_status': data_status,
            'loading': False
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_initial_data: {e}")
        return jsonify({
            'error': 'Failed to load initial data',
            'message': str(e)
        }), 500

@api_bp.route('/orders', methods=['GET'])
def get_orders():
    """Get orders data with filtering and pagination"""
    try:
        # Parse query parameters
        city_id = request.args.get('city_id', type=int)
        business_line = request.args.get('business_line')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', type=int, default=1000)
        offset = request.args.get('offset', type=int, default=0)
        use_cache = request.args.get('use_cache', 'true').lower() == 'true'
        
        # Validate parameters
        if limit > 10000:  # Prevent excessive data requests
            return jsonify({
                'error': 'Limit too large',
                'message': 'Maximum limit is 10,000 records'
            }), 400
        
        data_service = get_data_service()
        
        # For async data service, we need to run in event loop
        async def fetch_orders():
            return await data_service.get_orders_data(
                city_id=city_id,
                business_line=business_line,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                offset=offset,
                use_cache=use_cache
            )
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(fetch_orders())
        finally:
            loop.close()
        
        if not result.success:
            return jsonify({
                'error': 'Failed to fetch orders data',
                'message': result.error
            }), 500
        
        # Convert DataFrame to records
        if result.data is not None and not result.data.empty:
            records = result.data.to_dict('records')
            
            # Add metadata
            response_data = {
                'data': records,
                'metadata': {
                    'total_records': len(records),
                    'source': result.source.value if result.source else 'unknown',
                    'fetch_time': result.fetch_time,
                    'cached': result.cached,
                    'filters': {
                        'city_id': city_id,
                        'business_line': business_line,
                        'start_date': start_date,
                        'end_date': end_date,
                        'limit': limit,
                        'offset': offset
                    }
                }
            }
        else:
            response_data = {
                'data': [],
                'metadata': {
                    'total_records': 0,
                    'source': result.source.value if result.source else 'unknown',
                    'fetch_time': result.fetch_time,
                    'cached': result.cached,
                    'message': 'No data found matching the specified criteria'
                }
            }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in get_orders: {e}")
        return jsonify({
            'error': 'Failed to fetch orders',
            'message': str(e)
        }), 500

@api_bp.route('/vendors', methods=['GET'])
def get_vendors():
    """Get vendors data with filtering and pagination"""
    try:
        # Parse query parameters
        city_id = request.args.get('city_id', type=int)
        business_line = request.args.get('business_line')
        status_ids = [int(x) for x in request.args.getlist('status_ids') if x.isdigit()]
        grades = request.args.getlist('grades')
        limit = request.args.get('limit', type=int, default=1000)
        offset = request.args.get('offset', type=int, default=0)
        use_cache = request.args.get('use_cache', 'true').lower() == 'true'
        
        # Validate parameters
        if limit > 10000:
            return jsonify({
                'error': 'Limit too large',
                'message': 'Maximum limit is 10,000 records'
            }), 400
        
        data_service = get_data_service()
        
        async def fetch_vendors():
            return await data_service.get_vendors_data(
                city_id=city_id,
                business_line=business_line,
                status_ids=status_ids if status_ids else None,
                grades=grades if grades else None,
                limit=limit,
                offset=offset,
                use_cache=use_cache
            )
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(fetch_vendors())
        finally:
            loop.close()
        
        if not result.success:
            return jsonify({
                'error': 'Failed to fetch vendors data',
                'message': result.error
            }), 500
        
        # Convert DataFrame to records
        if result.data is not None and not result.data.empty:
            records = result.data.to_dict('records')
            
            response_data = {
                'data': records,
                'metadata': {
                    'total_records': len(records),
                    'source': result.source.value if result.source else 'unknown',
                    'fetch_time': result.fetch_time,
                    'cached': result.cached,
                    'filters': {
                        'city_id': city_id,
                        'business_line': business_line,
                        'status_ids': status_ids,
                        'grades': grades,
                        'limit': limit,
                        'offset': offset
                    }
                }
            }
        else:
            response_data = {
                'data': [],
                'metadata': {
                    'total_records': 0,
                    'source': result.source.value if result.source else 'unknown',
                    'fetch_time': result.fetch_time,
                    'cached': result.cached,
                    'message': 'No vendors found matching the specified criteria'
                }
            }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in get_vendors: {e}")
        return jsonify({
            'error': 'Failed to fetch vendors',
            'message': str(e)
        }), 500

@api_bp.route('/heatmap', methods=['GET'])
def generate_heatmap():
    """Generate heatmap data"""
    try:
        # Parse and validate parameters
        heatmap_type = request.args.get('type', 'order_density')
        city_id = request.args.get('city_id', type=int)
        business_line = request.args.get('business_line')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        zoom_level = request.args.get('zoom_level', type=int, default=11)
        aggregation = request.args.get('aggregation', 'count')
        max_points = request.args.get('max_points', type=int, default=5000)
        async_mode = request.args.get('async', 'false').lower() == 'true'
        
        # Validate required parameters
        if not city_id:
            return jsonify({
                'error': 'Missing required parameter',
                'message': 'city_id is required'
            }), 400
        
        try:
            heatmap_type_enum = HeatmapType(heatmap_type)
            aggregation_enum = AggregationMethod(aggregation)
        except ValueError as e:
            return jsonify({
                'error': 'Invalid parameter',
                'message': f'Invalid heatmap type or aggregation method: {e}'
            }), 400
        
        # Create heatmap request
        heatmap_request = HeatmapRequest(
            heatmap_type=heatmap_type_enum,
            city_id=city_id,
            business_line=business_line,
            start_date=start_date,
            end_date=end_date,
            zoom_level=zoom_level,
            aggregation_method=aggregation_enum,
            max_points=max_points
        )
        
        # Handle async mode (submit as background task)
        if async_mode:
            task = submit_background_task(
                'tapsi_food_map.generate_heatmap',
                heatmap_type=heatmap_type,
                city_id=city_id,
                business_line=business_line,
                start_date=start_date,
                end_date=end_date,
                zoom_level=zoom_level
            )
            
            if task:
                return jsonify({
                    'async': True,
                    'task_id': task.id,
                    'status': 'submitted',
                    'status_url': f'/api/task/{task.id}'
                }), 202
            else:
                # Fallback to synchronous if background tasks not available
                async_mode = False
        
        # Synchronous generation
        map_service = get_map_service()
        
        async def generate():
            return await map_service.generate_heatmap(heatmap_request)
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(generate())
        finally:
            loop.close()
        
        if not result.success:
            return jsonify({
                'error': 'Heatmap generation failed',
                'message': result.error
            }), 500
        
        return jsonify({
            'success': True,
            'points': result.points,
            'metadata': result.metadata,
            'generation_time': result.generation_time,
            'async': False
        }), 200
        
    except Exception as e:
        logger.error(f"Error in generate_heatmap: {e}")
        return jsonify({
            'error': 'Heatmap generation error',
            'message': str(e)
        }), 500

@api_bp.route('/task/<task_id>', methods=['GET'])
def get_task_status_route(task_id: str):
    """Get status of background task"""
    try:
        status = get_task_status(task_id)
        return jsonify(status), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@api_bp.route('/aggregated', methods=['GET'])
def get_aggregated_data():
    """Get pre-aggregated data for performance"""
    try:
        aggregation_type = request.args.get('type', 'hourly')
        city_id = request.args.get('city_id', type=int)
        days_back = request.args.get('days_back', type=int, default=7)
        use_cache = request.args.get('use_cache', 'true').lower() == 'true'
        
        # Validate parameters
        if aggregation_type not in ['hourly', 'daily', 'weekly']:
            return jsonify({
                'error': 'Invalid aggregation type',
                'message': 'Must be one of: hourly, daily, weekly'
            }), 400
        
        if days_back > 90:  # Prevent excessive historical queries
            return jsonify({
                'error': 'Days back too large',
                'message': 'Maximum days_back is 90'
            }), 400
        
        data_service = get_data_service()
        
        async def fetch_aggregated():
            return await data_service.get_aggregated_data(
                aggregation_type=aggregation_type,
                city_id=city_id,
                days_back=days_back,
                use_cache=use_cache
            )
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(fetch_aggregated())
        finally:
            loop.close()
        
        if not result.success:
            return jsonify({
                'error': 'Failed to fetch aggregated data',
                'message': result.error
            }), 500
        
        # Convert DataFrame to records
        if result.data is not None and not result.data.empty:
            records = result.data.to_dict('records')
            
            response_data = {
                'data': records,
                'metadata': {
                    'aggregation_type': aggregation_type,
                    'city_id': city_id,
                    'days_back': days_back,
                    'total_records': len(records),
                    'source': result.source.value if result.source else 'unknown',
                    'fetch_time': result.fetch_time,
                    'cached': result.cached
                }
            }
        else:
            response_data = {
                'data': [],
                'metadata': {
                    'aggregation_type': aggregation_type,
                    'city_id': city_id,
                    'days_back': days_back,
                    'total_records': 0,
                    'message': 'No aggregated data found'
                }
            }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in get_aggregated_data: {e}")
        return jsonify({
            'error': 'Failed to fetch aggregated data',
            'message': str(e)
        }), 500

@api_bp.route('/cache/invalidate', methods=['POST'])
def invalidate_cache():
    """Invalidate cache entries (admin endpoint)"""
    try:
        data = request.get_json()
        if not data or 'pattern' not in data:
            return jsonify({
                'error': 'Missing pattern',
                'message': 'Request body must contain "pattern" field'
            }), 400
        
        pattern = data['pattern']
        data_type = data.get('data_type')
        
        cache_service = get_cache_service()
        count = cache_service.invalidate_pattern(pattern, data_type)
        
        return jsonify({
            'success': True,
            'invalidated_keys': count,
            'pattern': pattern
        }), 200
        
    except Exception as e:
        logger.error(f"Error in invalidate_cache: {e}")
        return jsonify({
            'error': 'Cache invalidation failed',
            'message': str(e)
        }), 500