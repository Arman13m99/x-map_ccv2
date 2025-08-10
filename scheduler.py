# scheduler.py - APScheduler for automated data fetching
import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from threading import Lock

from config import get_config
from mini import fetch_question_data

logger = logging.getLogger(__name__)

class DataScheduler:
    """Background scheduler for automated data fetching"""
    
    def __init__(self):
        self.config = get_config()
        self.scheduler = None
        self._data_lock = Lock()
        
        # Global data storage
        self.df_orders = None
        self.df_vendors = None
        self.gdf_marketing_areas = {}
        self.gdf_tehran_region = None
        self.gdf_tehran_main_districts = None
        self.df_coverage_targets = None
        self.target_lookup_dict = {}
        
        # Data freshness tracking
        self.last_order_fetch = None
        self.last_vendor_fetch = None
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 60  # seconds
        
    def start(self):
        """Start the background scheduler"""
        if self.scheduler is not None:
            logger.warning("Scheduler already running")
            return
            
        # Configure scheduler
        jobstores = {'default': MemoryJobStore()}
        executors = {
            'default': ThreadPoolExecutor(max_workers=self.config.scheduler.executors['default']['max_workers'])
        }
        job_defaults = self.config.scheduler.job_defaults
        
        self.scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=self.config.scheduler.timezone
        )
        
        # Add event listeners
        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
        
        # Schedule jobs
        self._schedule_jobs()
        
        # Start scheduler
        self.scheduler.start()
        logger.info("Data scheduler started successfully")
        
        # Trigger initial data load asynchronously
        self.scheduler.add_job(
            func=self._initial_data_load,
            trigger='date',
            run_date=datetime.now(),
            id='initial_load',
            max_instances=1
        )
    
    def stop(self):
        """Stop the background scheduler"""
        if self.scheduler is not None:
            self.scheduler.shutdown(wait=True)
            self.scheduler = None
            logger.info("Data scheduler stopped")
    
    def _schedule_jobs(self):
        """Schedule background jobs"""
        
        # Daily order data fetch at 9 AM UTC
        self.scheduler.add_job(
            func=self._fetch_order_data,
            trigger=IntervalTrigger(minutes=1),
            id='daily_order_fetch',
            name='Daily Order Data Fetch',
            max_instances=1,
            replace_existing=True
        )
        
        # Vendor data fetch every 10 minutes
        self.scheduler.add_job(
            func=self._fetch_vendor_data,
            trigger=IntervalTrigger(minutes=1),
            id='vendor_data_fetch',
            name='Vendor Data Fetch (10min)',
            max_instances=1,
            replace_existing=True
        )
        
        # Static data reload every hour
        self.scheduler.add_job(
            func=self._load_static_data,
            trigger=IntervalTrigger(minutes=1),
            id='static_data_reload',
            name='Static Data Reload (1hr)',
            max_instances=1,
            replace_existing=True
        )
        
        logger.info("Background jobs scheduled successfully")
    
    def _initial_data_load(self):
        """Initial data load on startup"""
        logger.info("🚀 Starting initial data load...")
        try:
            # Load static data first (polygons, targets)
            self._load_static_data()
            
            # Then load dynamic data (orders, vendors)
            self._fetch_order_data()
            self._fetch_vendor_data()
            
            logger.info("✅ Initial data load completed")
        except Exception as e:
            logger.error(f"❌ Initial data load failed: {e}", exc_info=True)
    
    def _fetch_order_data(self):
        """Fetch order data from Metabase"""
        logger.info("🔄 Fetching order data...")
        start_time = time.time()
        
        try:
            # Fetch data with optimized settings
            df_orders = fetch_question_data(
                question_id=self.config.metabase.order_data_question_id,
                metabase_url=self.config.metabase.url,
                username=self.config.metabase.username,
                password=self.config.metabase.password,
                team=self.config.metabase.team,
                workers=self.config.app.worker_count,
                page_size=self.config.app.page_size
            )
            
            if df_orders is None or df_orders.empty:
                logger.error("❌ No order data received from Metabase")
                return
            
            # Optimize dtypes for memory efficiency
            dtype_dict = {
                'city_id': 'Int64',
                'business_line': 'category',
                'marketing_area': 'category',
                'vendor_code': 'str',
                'organic': 'int8'
            }
            
            # Apply dtypes where columns exist
            for col, dtype in dtype_dict.items():
                if col in df_orders.columns:
                    try:
                        df_orders[col] = df_orders[col].astype(dtype)
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to {dtype}: {e}")
            
            # Process datetime columns
            if 'created_at' in df_orders.columns:
                df_orders['created_at'] = pd.to_datetime(df_orders['created_at'], errors='coerce')
                df_orders['created_at'] = df_orders['created_at'].dt.tz_localize(None)
            
            # Add city names
            if 'city_id' in df_orders.columns:
                df_orders['city_name'] = df_orders['city_id'].map(self.config.app.city_id_map).astype('category')
            
            # Add organic column if missing (for demo purposes)
            if 'organic' not in df_orders.columns:
                df_orders['organic'] = np.random.choice([0, 1], size=len(df_orders), p=[0.7, 0.3]).astype('int8')
            
            # Thread-safe update
            with self._data_lock:
                self.df_orders = df_orders
                self.last_order_fetch = datetime.now()
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Order data loaded: {len(df_orders):,} rows in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error fetching order data: {e}", exc_info=True)
            self._handle_fetch_error('order_data', e)
    
    def _fetch_vendor_data(self):
        """Fetch vendor data from Metabase"""
        logger.info("🔄 Fetching vendor data...")
        start_time = time.time()
        
        try:
            # Fetch live vendor data
            df_vendors_raw = fetch_question_data(
                question_id=self.config.metabase.vendor_data_question_id,
                metabase_url=self.config.metabase.url,
                username=self.config.metabase.username,
                password=self.config.metabase.password,
                team=self.config.metabase.team,
                workers=self.config.app.worker_count,
                page_size=self.config.app.page_size
            )
            
            if df_vendors_raw is None or df_vendors_raw.empty:
                logger.error("❌ No vendor data received from Metabase")
                return
            
            # Optimize dtypes
            vendor_dtype = {
                'city_id': 'Int64',
                'vendor_code': 'str',
                'status_id': 'float32',
                'visible': 'float32',
                'open': 'float32',
                'radius': 'float32'
            }
            
            for col, dtype in vendor_dtype.items():
                if col in df_vendors_raw.columns:
                    try:
                        df_vendors_raw[col] = df_vendors_raw[col].astype(dtype)
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to {dtype}: {e}")
            
            # Add city names
            if 'city_id' in df_vendors_raw.columns:
                df_vendors_raw['city_name'] = df_vendors_raw['city_id'].map(self.config.app.city_id_map).astype('category')
            
            # Merge with grades if available
            df_vendors = self._merge_vendor_grades(df_vendors_raw)
            
            # Add business line from orders
            df_vendors = self._add_vendor_business_lines(df_vendors)
            
            # Ensure required columns exist
            required_cols = ['latitude', 'longitude', 'vendor_name', 'radius', 'status_id', 'visible', 'open', 'vendor_code']
            for col in required_cols:
                if col not in df_vendors.columns:
                    df_vendors[col] = np.nan
            
            # Convert numeric columns
            numeric_cols = ['visible', 'open', 'status_id']
            for col in numeric_cols:
                if col in df_vendors.columns:
                    df_vendors[col] = pd.to_numeric(df_vendors[col], errors='coerce')
            
            # Ensure vendor_code is string
            if 'vendor_code' in df_vendors.columns:
                df_vendors['vendor_code'] = df_vendors['vendor_code'].astype(str)
            
            # Store original radius for reset functionality
            if 'radius' in df_vendors.columns:
                df_vendors['original_radius'] = df_vendors['radius'].copy()
            
            # Thread-safe update
            with self._data_lock:
                self.df_vendors = df_vendors
                self.last_vendor_fetch = datetime.now()
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Vendor data loaded: {len(df_vendors):,} rows in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error fetching vendor data: {e}", exc_info=True)
            self._handle_fetch_error('vendor_data', e)
    
    def _merge_vendor_grades(self, df_vendors_raw):
        """Merge vendor data with grades from CSV file"""
        try:
            graded_file = os.path.join(self.config.app.src_dir, 'vendor', 'graded.csv')
            if not os.path.exists(graded_file):
                logger.warning(f"Graded file not found: {graded_file}")
                df_vendors_raw['grade'] = pd.Categorical(['Ungraded'] * len(df_vendors_raw))
                return df_vendors_raw
            
            df_graded_data = pd.read_csv(graded_file, dtype={'vendor_code': 'str', 'grade': 'category'})
            
            if 'vendor_code' in df_vendors_raw.columns and 'vendor_code' in df_graded_data.columns:
                df_vendors_raw['vendor_code'] = df_vendors_raw['vendor_code'].astype(str)
                df_graded_data['vendor_code'] = df_graded_data['vendor_code'].astype(str)
                
                df_vendors = pd.merge(
                    df_vendors_raw, 
                    df_graded_data[['vendor_code', 'grade']], 
                    on='vendor_code', 
                    how='left'
                )
                
                if 'grade' in df_vendors.columns and pd.api.types.is_categorical_dtype(df_vendors['grade']):
                    df_vendors['grade'] = df_vendors['grade'].cat.add_categories(['Ungraded'])
                
                df_vendors['grade'] = df_vendors['grade'].fillna('Ungraded').astype('category')
                logger.info(f"Grades loaded and merged. Found {df_vendors['grade'].notna().sum()} graded vendors.")
                
                return df_vendors
            else:
                logger.warning("vendor_code column missing, grades not merged")
                df_vendors_raw['grade'] = pd.Categorical(['Ungraded'] * len(df_vendors_raw))
                return df_vendors_raw
                
        except Exception as e:
            logger.error(f"Error merging grades: {e}")
            df_vendors_raw['grade'] = pd.Categorical(['Ungraded'] * len(df_vendors_raw))
            return df_vendors_raw
    
    def _add_vendor_business_lines(self, df_vendors):
        """Add business line to vendors by joining with orders"""
        try:
            with self._data_lock:
                if (self.df_orders is not None and not self.df_orders.empty and 
                    'vendor_code' in df_vendors.columns and 'vendor_code' in self.df_orders.columns):
                    
                    vendor_bl = self.df_orders.groupby('vendor_code')['business_line'].agg(
                        lambda x: x.mode()[0] if not x.empty and not x.mode().empty else np.nan
                    )
                    
                    df_vendors = df_vendors.merge(
                        vendor_bl.rename('business_line'), 
                        left_on='vendor_code', 
                        right_index=True, 
                        how='left'
                    )
                    
                    if 'business_line' in df_vendors.columns:
                        df_vendors['business_line'] = df_vendors['business_line'].astype('category')
                    
                    logger.info("Business lines added to vendors from order data")
            
            return df_vendors
            
        except Exception as e:
            logger.error(f"Error adding business lines to vendors: {e}")
            return df_vendors
    
    def _load_static_data(self):
        """Load static data (polygons, targets)"""
        logger.info("🔄 Loading static data...")
        start_time = time.time()
        
        try:
            import geopandas as gpd
            from shapely import wkt
            
            # Load marketing areas
            marketing_areas = {}
            cities = ['mashhad', 'shiraz', 'tehran']
            
            for city in cities:
                polygon_file = os.path.join(
                    self.config.app.src_dir, 
                    'polygons', 
                    'tapsifood_marketing_areas', 
                    f'{city}_polygons.csv'
                )
                
                if os.path.exists(polygon_file):
                    try:
                        df_poly = pd.read_csv(polygon_file)
                        if 'polygon' in df_poly.columns and 'marketing_area' in df_poly.columns:
                            df_poly['geometry'] = df_poly['polygon'].apply(wkt.loads)
                            gdf_poly = gpd.GeoDataFrame(df_poly, geometry='geometry')
                            marketing_areas[city] = gdf_poly
                            logger.info(f"Loaded {len(gdf_poly)} polygons for {city}")
                        else:
                            logger.warning(f"Required columns missing in {polygon_file}")
                    except Exception as e:
                        logger.error(f"Error loading {city} polygons: {e}")
                else:
                    logger.warning(f"Polygon file not found: {polygon_file}")
            
            # Load Tehran region data
            tehran_region = None
            tehran_districts = None
            
            try:
                region_shp = os.path.join(
                    self.config.app.src_dir, 
                    'polygons', 
                    'tehran_districts', 
                    'RegionTehran_WGS1984.shp'
                )
                if os.path.exists(region_shp):
                    tehran_region = gpd.read_file(region_shp)
                    logger.info(f"Loaded Tehran region data: {len(tehran_region)} regions")
            except Exception as e:
                logger.error(f"Error loading Tehran region data: {e}")
            
            try:
                districts_shp = os.path.join(
                    self.config.app.src_dir, 
                    'polygons', 
                    'tehran_districts', 
                    'Tehran_WGS1984.shp'
                )
                if os.path.exists(districts_shp):
                    tehran_districts = gpd.read_file(districts_shp)
                    logger.info(f"Loaded Tehran districts data: {len(tehran_districts)} districts")
            except Exception as e:
                logger.error(f"Error loading Tehran districts data: {e}")
            
            # Load coverage targets
            coverage_targets = None
            target_lookup = {}
            
            try:
                targets_file = os.path.join(
                    self.config.app.src_dir, 
                    'targets', 
                    'tehran_coverage.csv'
                )
                if os.path.exists(targets_file):
                    coverage_targets = pd.read_csv(targets_file)
                    if 'area_id' in coverage_targets.columns and 'coverage_target' in coverage_targets.columns:
                        target_lookup = dict(zip(coverage_targets['area_id'], coverage_targets['coverage_target']))
                        logger.info(f"Loaded coverage targets: {len(target_lookup)} areas")
                    else:
                        logger.warning(f"Required columns missing in {targets_file}")
                else:
                    logger.warning(f"Coverage targets file not found: {targets_file}")
            except Exception as e:
                logger.error(f"Error loading coverage targets: {e}")
            
            # Thread-safe update
            with self._data_lock:
                self.gdf_marketing_areas = marketing_areas
                self.gdf_tehran_region = tehran_region
                self.gdf_tehran_main_districts = tehran_districts
                self.df_coverage_targets = coverage_targets
                self.target_lookup_dict = target_lookup
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Static data loaded in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error loading static data: {e}", exc_info=True)
    
    def _handle_fetch_error(self, data_type: str, error: Exception):
        """Handle data fetching errors with retry logic"""
        logger.error(f"Error fetching {data_type}: {error}")
        
        # Schedule retry job
        retry_job_id = f"retry_{data_type}_{int(time.time())}"
        
        if data_type == 'order_data':
            retry_func = self._fetch_order_data
        elif data_type == 'vendor_data':
            retry_func = self._fetch_vendor_data
        else:
            return
        
        self.scheduler.add_job(
            func=retry_func,
            trigger='date',
            run_date=datetime.now().replace(second=datetime.now().second + self.retry_delay),
            id=retry_job_id,
            max_instances=1
        )
        
        logger.info(f"Scheduled retry for {data_type} in {self.retry_delay} seconds")
    
    def _job_executed(self, event):
        """Handle successful job execution"""
        logger.info(f"Job '{event.job_id}' executed successfully")
    
    def _job_error(self, event):
        """Handle job execution errors"""
        logger.error(f"Job '{event.job_id}' crashed: {event.exception}")
    
    def get_data_status(self) -> dict:
        """Get current data status and freshness"""
        with self._data_lock:
            return {
                'orders': {
                    'loaded': self.df_orders is not None,
                    'rows': len(self.df_orders) if self.df_orders is not None else 0,
                    'last_fetch': self.last_order_fetch.isoformat() if self.last_order_fetch else None
                },
                'vendors': {
                    'loaded': self.df_vendors is not None,
                    'rows': len(self.df_vendors) if self.df_vendors is not None else 0,
                    'last_fetch': self.last_vendor_fetch.isoformat() if self.last_vendor_fetch else None
                },
                'static_data': {
                    'marketing_areas': len(self.gdf_marketing_areas),
                    'tehran_region_loaded': self.gdf_tehran_region is not None,
                    'tehran_districts_loaded': self.gdf_tehran_main_districts is not None,
                    'coverage_targets': len(self.target_lookup_dict)
                }
            }
    
    def get_orders_data(self):
        """Thread-safe access to orders data"""
        with self._data_lock:
            return self.df_orders.copy() if self.df_orders is not None else None
    
    def get_vendors_data(self):
        """Thread-safe access to vendors data"""
        with self._data_lock:
            return self.df_vendors.copy() if self.df_vendors is not None else None
    
    def get_static_data(self):
        """Thread-safe access to static data"""
        with self._data_lock:
            return {
                'marketing_areas': self.gdf_marketing_areas.copy(),
                'tehran_region': self.gdf_tehran_region.copy() if self.gdf_tehran_region is not None else None,
                'tehran_districts': self.gdf_tehran_main_districts.copy() if self.gdf_tehran_main_districts is not None else None,
                'coverage_targets': self.df_coverage_targets.copy() if self.df_coverage_targets is not None else None,
                'target_lookup': self.target_lookup_dict.copy()
            }

# Global scheduler instance
data_scheduler = DataScheduler()

def get_scheduler() -> DataScheduler:
    """Get the global scheduler instance"""
    return data_scheduler