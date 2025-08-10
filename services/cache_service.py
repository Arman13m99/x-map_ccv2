# services/cache_service.py - High-level caching operations
import logging
import time
import json
import hashlib
from typing import Any, Optional, List, Dict, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps
import asyncio

from cache import get_cache_manager

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Caching strategies for different data types"""
    WRITE_THROUGH = "write_through"      # Write to cache and storage
    WRITE_AROUND = "write_around"        # Write to storage, invalidate cache
    WRITE_BACK = "write_back"           # Write to cache, async write to storage
    READ_THROUGH = "read_through"        # Read from cache, fallback to storage
    CACHE_ASIDE = "cache_aside"         # Manual cache management

class CacheTier(Enum):
    """Cache tiers with different TTLs and priorities"""
    HOT = "hot"           # Frequently accessed data (short TTL)
    WARM = "warm"         # Moderately accessed data (medium TTL)
    COLD = "cold"         # Infrequently accessed data (long TTL)
    PERSISTENT = "persistent"  # Long-term cached data

@dataclass
class CachePolicy:
    """Cache policy configuration"""
    strategy: CacheStrategy
    tier: CacheTier
    ttl: int
    max_size: Optional[int] = None
    compression: bool = False
    encryption: bool = False

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    average_access_time: float = 0.0
    hit_rate: float = 0.0

class CacheService:
    """High-level caching service with intelligent cache management"""
    
    def __init__(self):
        self.cache_manager = get_cache_manager()
        
        # Cache policies for different data types
        self.policies = {
            'heatmap': CachePolicy(
                strategy=CacheStrategy.CACHE_ASIDE,
                tier=CacheTier.WARM,
                ttl=1800,  # 30 minutes
                compression=True
            ),
            'orders': CachePolicy(
                strategy=CacheStrategy.READ_THROUGH,
                tier=CacheTier.HOT,
                ttl=900,   # 15 minutes
                compression=True
            ),
            'vendors': CachePolicy(
                strategy=CacheStrategy.READ_THROUGH,
                tier=CacheTier.WARM,
                ttl=600,   # 10 minutes
                compression=False
            ),
            'aggregated': CachePolicy(
                strategy=CacheStrategy.CACHE_ASIDE,
                tier=CacheTier.COLD,
                ttl=3600,  # 1 hour
                compression=True
            ),
            'static': CachePolicy(
                strategy=CacheStrategy.CACHE_ASIDE,
                tier=CacheTier.PERSISTENT,
                ttl=86400,  # 24 hours
                compression=False
            )
        }
        
        # Performance tracking
        self.metrics = {
            data_type: CacheMetrics() 
            for data_type in self.policies.keys()
        }
        
        self.global_metrics = CacheMetrics()
    
    async def get_with_fallback(self, 
                              key: str, 
                              fallback_func: Callable,
                              data_type: str = 'default',
                              **fallback_kwargs) -> Any:
        """Get data from cache with automatic fallback to data source"""
        start_time = time.time()
        
        try:
            # Try cache first
            cached_data = self.cache_manager.get(key)
            if cached_data is not None:
                self._update_metrics(data_type, hit=True, access_time=time.time() - start_time)
                logger.debug(f"Cache hit for key: {key}")
                return cached_data
            
            # Cache miss - execute fallback function
            logger.debug(f"Cache miss for key: {key}")
            
            if asyncio.iscoroutinefunction(fallback_func):
                data = await fallback_func(**fallback_kwargs)
            else:
                data = fallback_func(**fallback_kwargs)
            
            # Store in cache if data is not None
            if data is not None:
                policy = self.policies.get(data_type, self.policies['static'])
                success = self.cache_manager.set(key, data, ttl=policy.ttl)
                if not success:
                    logger.warning(f"Failed to cache data for key: {key}")
            
            self._update_metrics(data_type, hit=False, access_time=time.time() - start_time)
            return data
            
        except Exception as e:
            logger.error(f"Cache operation failed for key {key}: {e}")
            # Try fallback function on cache error
            try:
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(**fallback_kwargs)
                else:
                    return fallback_func(**fallback_kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback function failed: {fallback_error}")
                return None
    
    def set_with_policy(self, 
                       key: str, 
                       value: Any, 
                       data_type: str = 'default',
                       override_ttl: Optional[int] = None) -> bool:
        """Set cache value with appropriate policy"""
        try:
            policy = self.policies.get(data_type, self.policies['static'])
            ttl = override_ttl if override_ttl is not None else policy.ttl
            
            # Apply compression if configured
            if policy.compression and hasattr(value, '__dict__'):
                # For complex objects, could implement compression here
                pass
            
            success = self.cache_manager.set(key, value, ttl=ttl)
            
            if success:
                logger.debug(f"Cached data for key: {key} with TTL: {ttl}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to set cache for key {key}: {e}")
            return False
    
    def invalidate_pattern(self, pattern: str, data_type: Optional[str] = None) -> int:
        """Invalidate cache keys matching pattern"""
        try:
            count = self.cache_manager.clear_pattern(pattern)
            
            if data_type and data_type in self.metrics:
                self.metrics[data_type].evictions += count
            
            logger.info(f"Invalidated {count} cache keys matching pattern: {pattern}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to invalidate pattern {pattern}: {e}")
            return 0
    
    def warm_cache_async(self, 
                        warming_functions: List[Dict[str, Any]],
                        batch_size: int = 5) -> None:
        """Warm cache with multiple data sources asynchronously"""
        async def _warm_cache():
            try:
                # Process in batches to avoid overwhelming the system
                for i in range(0, len(warming_functions), batch_size):
                    batch = warming_functions[i:i+batch_size]
                    
                    tasks = []
                    for func_config in batch:
                        func = func_config.get('function')
                        key = func_config.get('key')
                        data_type = func_config.get('data_type', 'default')
                        kwargs = func_config.get('kwargs', {})
                        
                        if func and key:
                            task = self.get_with_fallback(
                                key=key,
                                fallback_func=func,
                                data_type=data_type,
                                **kwargs
                            )
                            tasks.append(task)
                    
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        successful = sum(1 for r in results if not isinstance(r, Exception))
                        logger.info(f"Cache warming batch completed: {successful}/{len(tasks)} successful")
                
                logger.info("Cache warming completed")
                
            except Exception as e:
                logger.error(f"Cache warming failed: {e}")
        
        # Schedule the warming task
        asyncio.create_task(_warm_cache())
    
    def create_cache_key(self, 
                        prefix: str, 
                        params: Dict[str, Any],
                        include_timestamp: bool = False) -> str:
        """Create consistent cache key from parameters"""
        try:
            # Filter out None values and convert to strings
            filtered_params = {}
            for k, v in params.items():
                if v is not None:
                    if isinstance(v, (list, dict)):
                        filtered_params[k] = json.dumps(v, sort_keys=True)
                    else:
                        filtered_params[k] = str(v)
            
            # Sort parameters for consistency
            param_string = '|'.join([f"{k}:{v}" for k, v in sorted(filtered_params.items())])
            
            # Add timestamp if requested (for time-sensitive caches)
            if include_timestamp:
                # Round to nearest 5 minutes for grouping
                timestamp = int(time.time() // 300) * 300
                param_string += f"|ts:{timestamp}"
            
            # Create hash for long keys
            if len(param_string) > 200:
                param_hash = hashlib.md5(param_string.encode()).hexdigest()
                return f"{prefix}:{param_hash}"
            
            return f"{prefix}:{param_string}"
            
        except Exception as e:
            logger.error(f"Failed to create cache key: {e}")
            # Fallback to simple key
            return f"{prefix}:{int(time.time())}"
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            # Get underlying cache stats
            cache_stats = self.cache_manager.get_stats()
            
            # Calculate per-data-type metrics
            data_type_stats = {}
            for data_type, metrics in self.metrics.items():
                total_requests = metrics.hits + metrics.misses
                hit_rate = (metrics.hits / total_requests * 100) if total_requests > 0 else 0
                
                data_type_stats[data_type] = {
                    'hits': metrics.hits,
                    'misses': metrics.misses,
                    'hit_rate': round(hit_rate, 2),
                    'evictions': metrics.evictions,
                    'average_access_time': round(metrics.average_access_time, 4)
                }
            
            # Global metrics
            total_hits = sum(m.hits for m in self.metrics.values())
            total_misses = sum(m.misses for m in self.metrics.values())
            total_requests = total_hits + total_misses
            global_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'cache_engine_stats': cache_stats,
                'global_metrics': {
                    'total_requests': total_requests,
                    'total_hits': total_hits,
                    'total_misses': total_misses,
                    'global_hit_rate': round(global_hit_rate, 2)
                },
                'by_data_type': data_type_stats,
                'policies': {
                    data_type: {
                        'strategy': policy.strategy.value,
                        'tier': policy.tier.value,
                        'ttl': policy.ttl,
                        'compression': policy.compression
                    }
                    for data_type, policy in self.policies.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {'error': str(e)}
    
    def optimize_cache_policies(self) -> Dict[str, Any]:
        """Analyze cache performance and suggest optimizations"""
        try:
            optimizations = {}
            
            for data_type, metrics in self.metrics.items():
                if metrics.hits + metrics.misses == 0:
                    continue
                
                hit_rate = metrics.hits / (metrics.hits + metrics.misses) * 100
                current_policy = self.policies[data_type]
                
                suggestions = []
                
                # Analyze hit rate
                if hit_rate < 30:
                    suggestions.append({
                        'type': 'increase_ttl',
                        'current_ttl': current_policy.ttl,
                        'suggested_ttl': min(current_policy.ttl * 2, 86400),
                        'reason': f'Low hit rate ({hit_rate:.1f}%)'
                    })
                elif hit_rate > 90 and current_policy.ttl > 300:
                    suggestions.append({
                        'type': 'decrease_ttl',
                        'current_ttl': current_policy.ttl,
                        'suggested_ttl': max(current_policy.ttl // 2, 300),
                        'reason': f'Very high hit rate ({hit_rate:.1f}%), can reduce TTL'
                    })
                
                # Analyze access time
                if metrics.average_access_time > 0.1:  # 100ms
                    suggestions.append({
                        'type': 'enable_compression',
                        'reason': f'High access time ({metrics.average_access_time:.3f}s)'
                    })
                
                # Analyze eviction rate
                total_operations = metrics.hits + metrics.misses
                eviction_rate = metrics.evictions / total_operations * 100 if total_operations > 0 else 0
                
                if eviction_rate > 10:
                    suggestions.append({
                        'type': 'increase_cache_size',
                        'eviction_rate': eviction_rate,
                        'reason': f'High eviction rate ({eviction_rate:.1f}%)'
                    })
                
                if suggestions:
                    optimizations[data_type] = {
                        'current_metrics': {
                            'hit_rate': round(hit_rate, 2),
                            'average_access_time': round(metrics.average_access_time, 4),
                            'eviction_rate': round(eviction_rate, 2)
                        },
                        'suggestions': suggestions
                    }
            
            return {
                'analysis_timestamp': datetime.now().isoformat(),
                'optimizations': optimizations
            }
            
        except Exception as e:
            logger.error(f"Cache optimization analysis failed: {e}")
            return {'error': str(e)}
    
    def _update_metrics(self, 
                       data_type: str, 
                       hit: bool, 
                       access_time: float) -> None:
        """Update cache metrics"""
        try:
            if data_type not in self.metrics:
                self.metrics[data_type] = CacheMetrics()
            
            metrics = self.metrics[data_type]
            
            if hit:
                metrics.hits += 1
            else:
                metrics.misses += 1
            
            # Update average access time (exponential moving average)
            alpha = 0.1  # Smoothing factor
            metrics.average_access_time = (
                alpha * access_time + 
                (1 - alpha) * metrics.average_access_time
            )
            
            # Update hit rate
            total_requests = metrics.hits + metrics.misses
            metrics.hit_rate = (metrics.hits / total_requests * 100) if total_requests > 0 else 0
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")

# Decorators for easy cache integration
class CacheDecorator:
    """Decorator for automatic caching"""
    
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
    
    def cached(self, 
              data_type: str = 'default',
              ttl: Optional[int] = None,
              key_func: Optional[Callable] = None):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self.cache_service.create_cache_key(
                        f"func_{func.__name__}",
                        {'args': args, 'kwargs': kwargs}
                    )
                
                # Get with fallback
                result = await self.cache_service.get_with_fallback(
                    key=cache_key,
                    fallback_func=func,
                    data_type=data_type,
                    *args,
                    **kwargs
                )
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self.cache_service.create_cache_key(
                        f"func_{func.__name__}",
                        {'args': args, 'kwargs': kwargs}
                    )
                
                # Simple sync cache check
                cached_result = self.cache_service.cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                if result is not None:
                    self.cache_service.set_with_policy(
                        cache_key, 
                        result, 
                        data_type,
                        override_ttl=ttl
                    )
                
                return result
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def cache_invalidate(self, pattern: str, data_type: Optional[str] = None):
        """Decorator for cache invalidation"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                self.cache_service.invalidate_pattern(pattern, data_type)
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                self.cache_service.invalidate_pattern(pattern, data_type)
                return result
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator

# Global cache service instance
cache_service = CacheService()
cache_decorator = CacheDecorator(cache_service)

def get_cache_service() -> CacheService:
    """Get the global cache service instance"""
    return cache_service

def get_cache_decorator() -> CacheDecorator:
    """Get the global cache decorator instance"""
    return cache_decorator