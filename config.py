# config.py - Environment-based Configuration Management
import os
import logging
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    url: str
    username: str
    password: str
    name: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass
class CacheConfig:
    """Cache configuration (Redis)"""
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    decode_responses: bool = True
    max_connections: int = 50

@dataclass
class MetabaseConfig:
    """Metabase API configuration"""
    url: str
    username: str
    password: str
    order_data_question_id: int
    vendor_data_question_id: int
    team: str = "growth"
    timeout: int = 300

@dataclass
class SchedulerConfig:
    """Background scheduler configuration"""
    timezone: str = "UTC"
    job_defaults: dict = None
    executors: dict = None
    
    def __post_init__(self):
        if self.job_defaults is None:
            self.job_defaults = {
                'coalesce': True,
                'max_instances': 1
            }
        if self.executors is None:
            self.executors = {
                'default': {'type': 'threadpool', 'max_workers': 20}
            }

@dataclass
class AppConfig:
    """Main application configuration"""
    # Flask settings
    debug: bool = False
    testing: bool = False
    secret_key: str = os.urandom(32).hex()
    
    # Server settings
    host: str = '0.0.0.0'
    port: int = 5000
    
    # Data processing settings
    chunk_size: int = 100000
    max_memory_per_session: str = "500MB"
    cache_ttl: int = 3600
    
    # Performance settings
    worker_count: int = 10
    page_size: int = 100000
    
    # File paths
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    src_dir: str = os.path.join(base_dir, 'src')
    public_dir: str = os.path.join(base_dir, 'public')
    
    # City mapping
    city_id_map: dict = None
    
    def __post_init__(self):
        if self.city_id_map is None:
            self.city_id_map = {1: "mashhad", 2: "tehran", 5: "shiraz"}

class Config:
    """Central configuration class"""
    
    def __init__(self):
        self.app = self._load_app_config()
        self.database = self._load_database_config()
        self.cache = self._load_cache_config()
        self.metabase = self._load_metabase_config()
        self.scheduler = self._load_scheduler_config()
        
        # Setup logging
        self._setup_logging()
    
    def _load_app_config(self) -> AppConfig:
        """Load application configuration from environment"""
        return AppConfig(
            debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true',
            testing=os.getenv('TESTING', 'false').lower() == 'true',
            secret_key=os.getenv('SECRET_KEY', os.urandom(32).hex()),
            host=os.getenv('HOST', '0.0.0.0'),
            port=int(os.getenv('PORT', '5000')),
            chunk_size=int(os.getenv('CHUNK_SIZE', '100000')),
            cache_ttl=int(os.getenv('CACHE_TTL', '3600')),
            worker_count=int(os.getenv('WORKER_COUNT', '10')),
            page_size=int(os.getenv('PAGE_SIZE', '100000'))
        )
    
    def _load_database_config(self) -> Optional[DatabaseConfig]:
        """Load database configuration from environment"""
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            # Database is optional in Phase 1
            return None
            
        return DatabaseConfig(
            url=db_url,
            username=os.getenv('DB_USERNAME', 'tapsi_user'),
            password=os.getenv('DB_PASSWORD', ''),
            name=os.getenv('DB_NAME', 'tapsi_food_map'),
            pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '20')),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30')),
            pool_recycle=int(os.getenv('DB_POOL_RECYCLE', '3600'))
        )
    
    def _load_cache_config(self) -> CacheConfig:
        """Load cache configuration from environment"""
        return CacheConfig(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            db=int(os.getenv('REDIS_DB', '0')),
            password=os.getenv('REDIS_PASSWORD'),
            socket_timeout=int(os.getenv('REDIS_SOCKET_TIMEOUT', '30')),
            socket_connect_timeout=int(os.getenv('REDIS_SOCKET_CONNECT_TIMEOUT', '30')),
            max_connections=int(os.getenv('REDIS_MAX_CONNECTIONS', '50'))
        )
    
    def _load_metabase_config(self) -> MetabaseConfig:
        """Load Metabase configuration from environment"""
        return MetabaseConfig(
            url=os.getenv('METABASE_URL', 'https://metabase.ofood.cloud'),
            username=os.getenv('METABASE_USERNAME', 'a.mehmandoost@OFOOD.CLOUD'),
            password=os.getenv('METABASE_PASSWORD', 'Fff322666@'),
            order_data_question_id=int(os.getenv('ORDER_DATA_QUESTION_ID', '5822')),
            vendor_data_question_id=int(os.getenv('VENDOR_DATA_QUESTION_ID', '5045')),
            team=os.getenv('METABASE_TEAM', 'growth'),
            timeout=int(os.getenv('METABASE_TIMEOUT', '300'))
        )
    
    def _load_scheduler_config(self) -> SchedulerConfig:
        """Load scheduler configuration from environment"""
        return SchedulerConfig(
            timezone=os.getenv('SCHEDULER_TIMEZONE', 'UTC')
        )
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_format = os.getenv('LOG_FORMAT', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                # Add file handler if LOG_FILE is specified
                *([logging.FileHandler(os.getenv('LOG_FILE'))] 
                  if os.getenv('LOG_FILE') else [])
            ]
        )
    
    def validate(self) -> bool:
        """Validate critical configuration values"""
        errors = []
        
        # Check Metabase credentials
        if not self.metabase.username or not self.metabase.password:
            errors.append("Metabase credentials not configured (METABASE_USERNAME, METABASE_PASSWORD)")
        
        # Check required question IDs
        if not self.metabase.order_data_question_id or not self.metabase.vendor_data_question_id:
            errors.append("Metabase question IDs not configured")
        
        if errors:
            for error in errors:
                logging.error(f"Configuration error: {error}")
            return False
            
        return True

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance"""
    return config