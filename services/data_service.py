# services/data_service.py - Data fetching and processing logic
import logging
import time
import asyncio
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import geopandas as gpd
import numpy as np
from dataclasses import dataclass
from enum import Enum

from config import get_config
from database import get_db_manager
from cache import get_cache_manager
from models import DataProcessor, filter_by_city, filter_by_business_line, filter_by_date_range
from mini import fetch_question_data

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Data source options"""
    METABASE = "metabase"
    DATABASE = "database"
    CACHE = "cache"

@dataclass
class DataFetchResult:
    """Result of data fetching operation"""
    success: bool
    data: Optional[pd.DataFrame] = None
    source: Optional[DataSource] = None
    record_count: int = 0
    fetch_time: float = 0.0
    error: Optional[str] = None
    cached: bool = False

@dataclass
class DataQuality:
    """Data quality metrics"""
    total_records: int
    null_coordinates: int
    invalid_coordinates: int
    duplicate_records: int
    data_completeness: float
    quality_score: float

class DataService:
    """High-level data service with intelligent source selection and caching"""
    
    def __init__(self):
        self.config = get_config()
        self.db_manager = get_db_manager()
        self.cache_manager = get_cache_manager()
        self.data_processor = DataProcessor()
        
        # Performance tracking
        self.fetch_stats = {
            'metabase_calls': 0,
            'database_calls': 0,
            'cache_hits': 0,
            'total_fetch_time': 0.0
        }
    
    async def get_orders_data(self,
                            city_id: Optional[int] = None,
                            business_line: Optional[str] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            limit: Optional[int] = None,
                            offset: Optional[int] = None,
                            use_cache: bool = True) -> DataFetchResult:
        """
        Get orders data with intelligent source selection
        Priority: Cache -> Database -> Metabase
        """
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key('orders', {
                'city_id': city_id,
                'business_line': business_line,
                'start_date': start_date,
                'end_date': end_date,
                'limit': limit,
                'offset': offset
            })
            
            # Try cache first
            if use_cache:
                cached_data = self.cache_manager.get(cache_key)
                if cached_data is not None:
                    self.fetch_stats['cache_hits'] += 1
                    return DataFetchResult(
                        success=True,
                        data=cached_data,
                        source=DataSource.CACHE,
                        record_count=len(cached_data),
                        fetch_time=time.time() - start_time,
                        cached=True
                    )
            
            # Try database next
            if self.db_manager._connected:
                try:
                    start_dt = pd.to_datetime(start_date) if start_date else None
                    end_dt = pd.to_datetime(end_date) if end_date else None
                    
                    df = self.db_manager.get_orders_filtered(
                        city_id=city_id,
                        business_line=business_line,
                        start_date=start_dt,
                        end_date=end_dt,
                        limit=limit,
                        offset=offset
                    )
                    
                    if df is not None and not df.empty:
                        # Cache the result
                        if use_cache:
                            self.cache_manager.set(cache_key, df, ttl=1800)  # 30 minutes
                        
                        self.fetch_stats['database_calls'] += 1
                        return DataFetchResult(
                            success=True,
                            data=df,
                            source=DataSource.DATABASE,
                            record_count=len(df),
                            fetch_time=time.time() - start_time
                        )
                        
                except Exception as e:
                    logger.warning(f"Database query failed, falling back to Metabase: {e}")
            
            # Fallback to Metabase (full data, then filter)
            df = await self._fetch_from_metabase('orders')
            if df is None or df.empty:
                return DataFetchResult(
                    success=False,
                    error="No data available from any source"
                )
            
            # Apply filters
            df_filtered = self._apply_filters(df, {
                'city_id': city_id,
                'business_line': business_line,
                'start_date': start_date,
                'end_date': end_date,
                'limit': limit,
                'offset': offset
            })
            
            # Cache the filtered result
            if use_cache and len(df_filtered) < 100000:  # Don't cache huge results
                self.cache_manager.set(cache_key, df_filtered, ttl=900)  # 15 minutes
            
            self.fetch_stats['metabase_calls'] += 1
            return DataFetchResult(
                success=True,
                data=df_filtered,
                source=DataSource.METABASE,
                record_count=len(df_filtered),
                fetch_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Failed to get orders data: {e}")
            return DataFetchResult(
                success=False,
                error=str(e),
                fetch_time=time.time() - start_time
            )
    
    async def get_vendors_data(self,
                             city_id: Optional[int] = None,
                             status_ids: Optional[List[int]] = None,
                             grades: Optional[List[str]] = None,
                             business_line: Optional[str] = None,
                             limit: Optional[int] = None,
                             offset: Optional[int] = None,
                             use_cache: bool = True) -> DataFetchResult:
        """
        Get vendors data with intelligent source selection
        """
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key('vendors', {
                'city_id': city_id,
                'status_ids': status_ids,
                'grades': grades,
                'business_line': business_line,
                'limit': limit,
                'offset': offset
            })
            
            # Try cache first
            if use_cache:
                cached_data = self.cache_manager.get(cache_key)
                if cached_data is not None:
                    self.fetch_stats['cache_hits'] += 1
                    return DataFetchResult(
                        success=True,
                        data=cached_data,
                        source=DataSource.CACHE,
                        record_count=len(cached_data),
                        fetch_time=time.time() - start_time,
                        cached=True
                    )
            
            # Try database next
            if self.db_manager._connected:
                try:
                    df = self.db_manager.get_vendors_filtered(
                        city_id=city_id,
                        status_ids=status_ids,
                        grades=grades,
                        business_line=business_line,
                        limit=limit,
                        offset=offset
                    )
                    
                    if df is not None and not df.empty:
                        # Cache the result
                        if use_cache:
                            self.cache_manager.set(cache_key, df, ttl=600)  # 10 minutes
                        
                        self.fetch_stats['database_calls'] += 1
                        return DataFetchResult(
                            success=True,
                            data=df,
                            source=DataSource.DATABASE,
                            record_count=len(df),
                            fetch_time=time.time() - start_time
                        )
                        
                except Exception as e:
                    logger.warning(f"Database query failed for vendors: {e}")
            
            # Fallback to Metabase
            df = await self._fetch_from_metabase('vendors')
            if df is None or df.empty:
                return DataFetchResult(
                    success=False,
                    error="No vendor data available from any source"
                )
            
            # Apply filters
            df_filtered = self._apply_vendor_filters(df, {
                'city_id': city_id,
                'status_ids': status_ids,
                'grades': grades,
                'business_line': business_line,
                'limit': limit,
                'offset': offset
            })
            
            # Cache the filtered result
            if use_cache and len(df_filtered) < 50000:
                self.cache_manager.set(cache_key, df_filtered, ttl=600)
            
            self.fetch_stats['metabase_calls'] += 1
            return DataFetchResult(
                success=True,
                data=df_filtered,
                source=DataSource.METABASE,
                record_count=len(df_filtered),
                fetch_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Failed to get vendors data: {e}")
            return DataFetchResult(
                success=False,
                error=str(e),
                fetch_time=time.time() - start_time
            )
    
    async def get_aggregated_data(self,
                                aggregation_type: str = 'hourly',
                                city_id: Optional[int] = None,
                                days_back: int = 7,
                                use_cache: bool = True) -> DataFetchResult:
        """Get pre-aggregated data for dashboard performance"""
        start_time = time.time()
        
        try:
            cache_key = self._generate_cache_key('aggregated', {
                'type': aggregation_type,
                'city_id': city_id,
                'days_back': days_back
            })
            
            # Try cache first
            if use_cache:
                cached_data = self.cache_manager.get(cache_key)
                if cached_data is not None:
                    self.fetch_stats['cache_hits'] += 1
                    return DataFetchResult(
                        success=True,
                        data=cached_data,
                        source=DataSource.CACHE,
                        record_count=len(cached_data),
                        fetch_time=time.time() - start_time,
                        cached=True
                    )
            
            # Try database aggregation
            if self.db_manager._connected:
                try:
                    df = self.db_manager.get_aggregated_data(
                        aggregation_type=aggregation_type,
                        city_id=city_id,
                        days_back=days_back
                    )
                    
                    if df is not None and not df.empty:
                        # Cache for longer since aggregated data is expensive
                        if use_cache:
                            self.cache_manager.set(cache_key, df, ttl=3600)  # 1 hour
                        
                        self.fetch_stats['database_calls'] += 1
                        return DataFetchResult(
                            success=True,
                            data=df,
                            source=DataSource.DATABASE,
                            record_count=len(df),
                            fetch_time=time.time() - start_time
                        )
                        
                except Exception as e:
                    logger.warning(f"Database aggregation failed: {e}")
            
            # Fallback: get raw data and aggregate in memory
            orders_result = await self.get_orders_data(
                city_id=city_id,
                start_date=(datetime.now() - timedelta(days=days_back)).isoformat(),
                use_cache=use_cache
            )
            
            if not orders_result.success or orders_result.data is None:
                return DataFetchResult(
                    success=False,
                    error="No data available for aggregation"
                )
            
            # Perform in-memory aggregation
            df_aggregated = self._aggregate_in_memory(
                orders_result.data, 
                aggregation_type
            )
            
            # Cache the result
            if use_cache:
                self.cache_manager.set(cache_key, df_aggregated, ttl=1800)
            
            return DataFetchResult(
                success=True,
                data=df_aggregated,
                source=DataSource.METABASE,
                record_count=len(df_aggregated),
                fetch_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Failed to get aggregated data: {e}")
            return DataFetchResult(
                success=False,
                error=str(e),
                fetch_time=time.time() - start_time
            )
    
    async def _fetch_from_metabase(self, data_type: str) -> Optional[pd.DataFrame]:
        """Fetch data from Metabase API with retry logic"""
        try:
            question_id = (
                self.config.metabase.order_data_question_id 
                if data_type == 'orders' 
                else self.config.metabase.vendor_data_question_id
            )
            
            # Use asyncio to run the sync fetch_question_data in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(
                    executor,
                    fetch_question_data,
                    question_id,
                    self.config.metabase.url,
                    self.config.metabase.username,
                    self.config.metabase.password,
                    self.config.metabase.team,
                    self.config.app.worker_count,
                    self.config.app.page_size
                )
                
                df = await future
            
            if df is not None and not df.empty:
                # Apply data processing and optimization
                df = self.data_processor.optimize_dataframe_memory(df)
                df = self.data_processor.validate_coordinates(df)
                
                logger.info(f"Fetched {len(df)} records from Metabase for {data_type}")
                
                # Store in database if available
                if self.db_manager._connected:
                    try:
                        if data_type == 'orders':
                            self.db_manager.bulk_insert_orders(df)
                        else:
                            self.db_manager.bulk_insert_vendors(df)
                    except Exception as e:
                        logger.warning(f"Failed to store {data_type} in database: {e}")
                
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch {data_type} from Metabase: {e}")
            return None
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to DataFrame"""
        if df.empty:
            return df
        
        df_filtered = df.copy()
        
        # City filter
        if filters.get('city_id') is not None:
            df_filtered = filter_by_city(df_filtered, filters['city_id'])
        
        # Business line filter  
        if filters.get('business_line'):
            df_filtered = filter_by_business_line(df_filtered, filters['business_line'])
        
        # Date range filter
        if filters.get('start_date') or filters.get('end_date'):
            df_filtered = filter_by_date_range(
                df_filtered, 
                filters.get('start_date'),
                filters.get('end_date')
            )
        
        # Pagination
        if filters.get('offset'):
            df_filtered = df_filtered.iloc[filters['offset']:]
        
        if filters.get('limit'):
            df_filtered = df_filtered.head(filters['limit'])
        
        return df_filtered
    
    def _apply_vendor_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply vendor-specific filters"""
        if df.empty:
            return df
        
        df_filtered = df.copy()
        
        # City filter
        if filters.get('city_id') is not None:
            df_filtered = filter_by_city(df_filtered, filters['city_id'])
        
        # Business line filter
        if filters.get('business_line'):
            df_filtered = filter_by_business_line(df_filtered, filters['business_line'])
        
        # Status filter
        if filters.get('status_ids'):
            df_filtered = df_filtered[df_filtered['status_id'].isin(filters['status_ids'])]
        
        # Grades filter
        if filters.get('grades'):
            df_filtered = df_filtered[df_filtered['grade'].isin(filters['grades'])]
        
        # Pagination
        if filters.get('offset'):
            df_filtered = df_filtered.iloc[filters['offset']:]
        
        if filters.get('limit'):
            df_filtered = df_filtered.head(filters['limit'])
        
        return df_filtered
    
    def _aggregate_in_memory(self, df: pd.DataFrame, aggregation_type: str) -> pd.DataFrame:
        """Perform in-memory aggregation"""
        if df.empty or 'created_at' not in df.columns:
            return pd.DataFrame()
        
        # Ensure created_at is datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Define grouping frequency
        freq_map = {
            'hourly': 'H',
            'daily': 'D', 
            'weekly': 'W'
        }
        freq = freq_map.get(aggregation_type, 'H')
        
        # Group and aggregate
        df_agg = df.groupby([
            pd.Grouper(key='created_at', freq=freq),
            'city_id',
            'business_line'
        ]).agg({
            'vendor_code': 'count',
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        # Rename columns
        df_agg.columns = ['time_bucket', 'city_id', 'business_line', 'order_count', 'avg_lat', 'avg_lng']
        
        return df_agg
    
    def _generate_cache_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate consistent cache key"""
        # Filter out None values and sort for consistency
        filtered_params = {k: v for k, v in params.items() if v is not None}
        param_string = '|'.join([f"{k}:{v}" for k, v in sorted(filtered_params.items())])
        return f"{prefix}:{param_string}"
    
    async def batch_process_data(self, 
                               operations: List[Dict[str, Any]],
                               max_workers: int = 5) -> List[DataFetchResult]:
        """Process multiple data operations concurrently"""
        try:
            loop = asyncio.get_event_loop()
            
            # Create tasks for concurrent execution
            tasks = []
            for operation in operations:
                op_type = operation.get('type')
                op_params = operation.get('params', {})
                
                if op_type == 'orders':
                    task = self.get_orders_data(**op_params)
                elif op_type == 'vendors':
                    task = self.get_vendors_data(**op_params)
                elif op_type == 'aggregated':
                    task = self.get_aggregated_data(**op_params)
                else:
                    continue
                
                tasks.append(task)
            
            # Execute tasks concurrently
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions
                final_results = []
                for result in results:
                    if isinstance(result, Exception):
                        final_results.append(DataFetchResult(
                            success=False,
                            error=str(result)
                        ))
                    else:
                        final_results.append(result)
                
                return final_results
            
            return []
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return [DataFetchResult(success=False, error=str(e))]
    
    def analyze_data_quality(self, df: pd.DataFrame) -> DataQuality:
        """Analyze data quality metrics"""
        if df.empty:
            return DataQuality(
                total_records=0,
                null_coordinates=0,
                invalid_coordinates=0,
                duplicate_records=0,
                data_completeness=0.0,
                quality_score=0.0
            )
        
        try:
            total_records = len(df)
            
            # Check for null coordinates
            null_coords = 0
            if 'latitude' in df.columns and 'longitude' in df.columns:
                null_coords = df[['latitude', 'longitude']].isnull().any(axis=1).sum()
            
            # Check for invalid coordinates
            invalid_coords = 0
            if 'latitude' in df.columns and 'longitude' in df.columns:
                invalid_coords = (
                    (df['latitude'] < -90) | (df['latitude'] > 90) |
                    (df['longitude'] < -180) | (df['longitude'] > 180)
                ).sum()
            
            # Check for duplicates
            duplicate_records = df.duplicated().sum()
            
            # Calculate completeness (non-null values across all columns)
            completeness = (df.notna().sum().sum()) / (total_records * len(df.columns)) * 100
            
            # Calculate quality score
            coord_quality = 1 - (null_coords + invalid_coords) / total_records
            duplicate_quality = 1 - duplicate_records / total_records
            quality_score = (coord_quality * 0.4 + duplicate_quality * 0.2 + completeness/100 * 0.4) * 100
            
            return DataQuality(
                total_records=total_records,
                null_coordinates=null_coords,
                invalid_coordinates=invalid_coords,
                duplicate_records=duplicate_records,
                data_completeness=completeness,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Data quality analysis failed: {e}")
            return DataQuality(
                total_records=len(df),
                null_coordinates=0,
                invalid_coordinates=0,
                duplicate_records=0,
                data_completeness=0.0,
                quality_score=0.0
            )
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service performance statistics"""
        total_calls = sum(self.fetch_stats.values()) - self.fetch_stats['total_fetch_time']
        avg_fetch_time = (
            self.fetch_stats['total_fetch_time'] / total_calls 
            if total_calls > 0 else 0
        )
        
        return {
            'total_calls': total_calls,
            'cache_hit_rate': (
                self.fetch_stats['cache_hits'] / total_calls * 100 
                if total_calls > 0 else 0
            ),
            'average_fetch_time': avg_fetch_time,
            'metabase_calls': self.fetch_stats['metabase_calls'],
            'database_calls': self.fetch_stats['database_calls'],
            'cache_hits': self.fetch_stats['cache_hits']
        }

# Global data service instance
data_service = DataService()

def get_data_service() -> DataService:
    """Get the global data service instance"""
    return data_service