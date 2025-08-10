# cache.py - Redis connection and caching layer
import json
import pickle
import hashlib
import logging
import time
from functools import wraps
from typing import Any, Optional, Union, Dict, List
from datetime import datetime, timedelta

import redis
from config import get_config

logger = logging.getLogger(__name__)

class CacheManager:
    """Redis-based cache manager with intelligent caching strategies"""
    
    def __init__(self):
        self.config = get_config()
        self.redis_client = None
        self.fallback_cache = {}  # In-memory fallback
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'fallback_hits': 0
        }
        
        # Initialize Redis connection
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection with fallback handling"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.cache.host,
                port=self.config.cache.port,
                db=self.config.cache.db,
                password=self.config.cache.password,
                socket_timeout=self.config.cache.socket_timeout,
                socket_connect_timeout=self.config.cache.socket_connect_timeout,
                decode_responses=self.config.cache.decode_responses,
                max_connections=self.config.cache.max_connections,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
            self.redis_client = None
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate consistent cache key from parameters"""
        # Create a string representation of arguments
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        key_string = "|".join(key_parts)
        
        # Hash long keys for consistency
        if len(key_string) > 200:
            key_hash = hashlib.md5(key_string.encode()).hexdigest()
            return f"{prefix}:{key_hash}"
        
        return f"{prefix}:{key_string}"
    
    def get(self, key: str, default=None) -> Any:
        """Get value from cache with fallback handling"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value is not None:
                    self.cache_stats['hits'] += 1
                    try:
                        # Try JSON first (for simple data)
                        return json.loads(value)
                    except json.JSONDecodeError:
                        # Fall back to pickle (for complex objects)
                        return pickle.loads(value.encode('latin-1') if isinstance(value, str) else value)
                else:
                    self.cache_stats['misses'] += 1
            else:
                # Use in-memory fallback
                if key in self.fallback_cache:
                    entry = self.fallback_cache[key]
                    if entry['expires_at'] > datetime.now():
                        self.cache_stats['fallback_hits'] += 1
                        return entry['value']
                    else:
                        del self.fallback_cache[key]
                
                self.cache_stats['misses'] += 1
                
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.cache_stats['errors'] += 1
        
        return default
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache with TTL"""
        if ttl is None:
            ttl = self.config.app.cache_ttl
        
        try:
            if self.redis_client:
                # Try JSON encoding first (more efficient)
                try:
                    encoded_value = json.dumps(value, default=str)
                    is_json = True
                except (TypeError, ValueError):
                    # Fall back to pickle for complex objects
                    encoded_value = pickle.dumps(value).decode('latin-1')
                    is_json = False
                
                success = self.redis_client.setex(key, ttl, encoded_value)
                
                # Store metadata about encoding type
                if success and not is_json:
                    self.redis_client.setex(f"{key}:pickle", ttl, "1")
                
                return bool(success)
            else:
                # Use in-memory fallback
                self.fallback_cache[key] = {
                    'value': value,
                    'expires_at': datetime.now() + timedelta(seconds=ttl)
                }
                return True
                
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.cache_stats['errors'] += 1
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if self.redis_client:
                deleted = self.redis_client.delete(key)
                # Also delete pickle metadata if exists
                self.redis_client.delete(f"{key}:pickle")
                return bool(deleted)
            else:
                if key in self.fallback_cache:
                    del self.fallback_cache[key]
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            self.cache_stats['errors'] += 1
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern (Redis only)"""
        try:
            if self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys:
                    return self.redis_client.delete(*keys)
            else:
                # Pattern matching for in-memory cache
                import fnmatch
                keys_to_delete = [k for k in self.fallback_cache.keys() 
                                if fnmatch.fnmatch(k, pattern)]
                for key in keys_to_delete:
                    del self.fallback_cache[key]
                return len(keys_to_delete)
                
            return 0
            
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            self.cache_stats['errors'] += 1
            return 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'errors': self.cache_stats['errors'],
            'hit_rate_percent': round(hit_rate, 2),
            'fallback_hits': self.cache_stats['fallback_hits'],
            'using_redis': self.redis_client is not None,
            'fallback_cache_size': len(self.fallback_cache)
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats.update({
                    'redis_memory_used': info.get('used_memory_human', 'N/A'),
                    'redis_connected_clients': info.get('connected_clients', 0),
                    'redis_hits': info.get('keyspace_hits', 0),
                    'redis_misses': info.get('keyspace_misses', 0)
                })
            except Exception:
                pass
        
        return stats
    
    def health_check(self) -> bool:
        """Check cache system health"""
        try:
            if self.redis_client:
                self.redis_client.ping()
                return True
            else:
                # In-memory cache is always "healthy"
                return True
        except Exception:
            return False

# Cache decorators for easy use
class CacheDecorator:
    """Decorator class for caching function results"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
    
    def cache_result(self, prefix: str = "func", ttl: int = None, key_func=None):
        """Decorator to cache function results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self.cache_manager._generate_cache_key(
                        f"{prefix}_{func.__name__}", *args, **kwargs
                    )
                
                # Try to get from cache
                result = self.cache_manager.get(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                if result is not None:
                    self.cache_manager.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def cache_heatmap(self, ttl: int = 3600):
        """Specialized decorator for heatmap caching"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create specific cache key for heatmap parameters
                params = {
                    'city_id': kwargs.get('city_id', args[0] if args else None),
                    'business_line': kwargs.get('business_line', args[1] if len(args) > 1 else None),
                    'data_type': kwargs.get('data_type', args[2] if len(args) > 2 else 'orders'),
                    'start_date': str(kwargs.get('start_date', args[3] if len(args) > 3 else '')),
                    'end_date': str(kwargs.get('end_date', args[4] if len(args) > 4 else ''))
                }
                
                cache_key = self.cache_manager._generate_cache_key('heatmap', **params)
                
                # Try to get from cache
                result = self.cache_manager.get(cache_key)
                if result is not None:
                    logger.info(f"Heatmap cache hit for key: {cache_key}")
                    return result
                
                # Execute function and cache result
                logger.info(f"Generating new heatmap for key: {cache_key}")
                result = func(*args, **kwargs)
                
                if result is not None and len(result) > 0:
                    self.cache_manager.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator

# Global cache manager instance
cache_manager = CacheManager()
cache_decorator = CacheDecorator(cache_manager)

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance"""
    return cache_manager

def get_cache_decorator() -> CacheDecorator:
    """Get the global cache decorator instance"""
    return cache_decorator

# Convenience functions for direct use
def cache_get(key: str, default=None) -> Any:
    """Get value from cache"""
    return cache_manager.get(key, default)

def cache_set(key: str, value: Any, ttl: int = None) -> bool:
    """Set value in cache"""
    return cache_manager.set(key, value, ttl)

def cache_delete(key: str) -> bool:
    """Delete value from cache"""
    return cache_manager.delete(key)

def cache_clear_pattern(pattern: str) -> int:
    """Clear all keys matching pattern"""
    return cache_manager.clear_pattern(pattern)

def warm_cache(data_scheduler):
    """Warm cache with common queries"""
    logger.info("🔥 Warming cache with common queries...")
    
    try:
        # Get data from scheduler
        orders_data = data_scheduler.get_orders_data()
        vendors_data = data_scheduler.get_vendors_data()
        
        if orders_data is None or vendors_data is None:
            logger.warning("Cannot warm cache - data not available")
            return
        
        # Cache common aggregations
        cities = orders_data['city_id'].unique() if 'city_id' in orders_data.columns else [2]
        business_lines = orders_data['business_line'].unique() if 'business_line' in orders_data.columns else [None]
        
        for city_id in cities:
            for business_line in business_lines:
                # Cache filtered data
                cache_key = cache_manager._generate_cache_key(
                    'filtered_orders', city_id=city_id, business_line=business_line
                )
                
                filtered_data = orders_data[
                    (orders_data['city_id'] == city_id) if 'city_id' in orders_data.columns else orders_data.index >= 0
                ]
                
                if business_line and 'business_line' in orders_data.columns:
                    filtered_data = filtered_data[filtered_data['business_line'] == business_line]
                
                cache_manager.set(cache_key, filtered_data, ttl=1800)  # 30 minutes
        
        logger.info("✅ Cache warmed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error warming cache: {e}")