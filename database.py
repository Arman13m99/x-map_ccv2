# database.py - PostgreSQL integration with connection pooling
import logging
import time
from typing import Optional, List, Dict, Any, Union
from contextlib import contextmanager
from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd
from dataclasses import dataclass

try:
    from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, DateTime, Float, Boolean
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.pool import QueuePool
    from sqlalchemy.exc import SQLAlchemyError, OperationalError
    from sqlalchemy.dialects.postgresql import insert
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from config import get_config

logger = logging.getLogger(__name__)

# Database Models
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class OrderRecord(Base):
        __tablename__ = 'orders'
        
        id = Column(Integer, primary_key=True)
        city_id = Column(Integer, nullable=False, index=True)
        vendor_code = Column(String(50), nullable=False, index=True)
        business_line = Column(String(100), nullable=True, index=True)
        marketing_area = Column(String(200), nullable=True, index=True)
        latitude = Column(Float, nullable=False)
        longitude = Column(Float, nullable=False)
        created_at = Column(DateTime, nullable=False, index=True)
        organic = Column(Integer, default=0)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    class VendorRecord(Base):
        __tablename__ = 'vendors'
        
        id = Column(Integer, primary_key=True)
        vendor_code = Column(String(50), unique=True, nullable=False, index=True)
        city_id = Column(Integer, nullable=False, index=True)
        latitude = Column(Float, nullable=False)
        longitude = Column(Float, nullable=False)
        status_id = Column(Float, nullable=True)
        visible = Column(Float, nullable=True)
        open = Column(Float, nullable=True)
        radius = Column(Float, nullable=True)
        vendor_name = Column(String(500), nullable=True)
        grade = Column(String(50), default='Ungraded', index=True)
        business_line = Column(String(100), nullable=True, index=True)
        original_radius = Column(Float, nullable=True)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    class DataFreshness(Base):
        __tablename__ = 'data_freshness'
        
        id = Column(Integer, primary_key=True)
        data_type = Column(String(50), unique=True, nullable=False)
        last_updated = Column(DateTime, nullable=False)
        record_count = Column(Integer, default=0)
        status = Column(String(20), default='success')
        error_message = Column(String(1000), nullable=True)
else:
    Base = None
    OrderRecord = None
    VendorRecord = None
    DataFreshness = None

@dataclass
class DatabaseStats:
    """Database statistics and health information"""
    connected: bool
    total_orders: int = 0
    total_vendors: int = 0
    last_order_update: Optional[datetime] = None
    last_vendor_update: Optional[datetime] = None
    connection_pool_size: int = 0
    active_connections: int = 0

class DatabaseManager:
    """PostgreSQL database manager with connection pooling and health monitoring"""
    
    def __init__(self):
        self.config = get_config()
        self.engine = None
        self.SessionLocal = None
        self._connected = False
        
        # Initialize if database is configured
        if self.config.database and SQLALCHEMY_AVAILABLE:
            self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection with pooling"""
        try:
            if not self.config.database:
                logger.warning("Database not configured, skipping initialization")
                return
            
            # Create engine with connection pooling
            self.engine = create_engine(
                self.config.database.url,
                poolclass=QueuePool,
                pool_size=self.config.database.pool_size,
                max_overflow=self.config.database.max_overflow,
                pool_timeout=self.config.database.pool_timeout,
                pool_recycle=self.config.database.pool_recycle,
                pool_pre_ping=True,  # Verify connections before use
                echo=self.config.app.debug,  # Log SQL in debug mode
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self._connected = True
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            self._connected = False
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        if not self._connected or not self.SessionLocal:
            yield None
            return
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self) -> bool:
        """Create database tables if they don't exist"""
        if not self._connected or not Base:
            logger.warning("Database not available for table creation")
            return False
        
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created/verified successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            return False
    
    def bulk_insert_orders(self, orders_df: pd.DataFrame) -> bool:
        """Bulk insert orders data with conflict resolution"""
        if not self._connected or orders_df.empty:
            return False
        
        try:
            # Prepare data for insertion
            orders_data = orders_df.to_dict('records')
            
            with self.get_session() as session:
                if session is None:
                    return False
                
                # Use PostgreSQL UPSERT (ON CONFLICT DO UPDATE)
                stmt = insert(OrderRecord.__table__).values(orders_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['city_id', 'vendor_code', 'created_at'],
                    set_=dict(
                        business_line=stmt.excluded.business_line,
                        marketing_area=stmt.excluded.marketing_area,
                        latitude=stmt.excluded.latitude,
                        longitude=stmt.excluded.longitude,
                        organic=stmt.excluded.organic,
                        updated_at=datetime.utcnow()
                    )
                )
                
                session.execute(stmt)
                
            # Update freshness tracking
            self._update_data_freshness('orders', len(orders_data))
            
            logger.info(f"Bulk inserted {len(orders_data)} order records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to bulk insert orders: {e}")
            return False
    
    def bulk_insert_vendors(self, vendors_df: pd.DataFrame) -> bool:
        """Bulk insert vendors data with conflict resolution"""
        if not self._connected or vendors_df.empty:
            return False
        
        try:
            # Prepare data for insertion
            vendors_data = vendors_df.to_dict('records')
            
            with self.get_session() as session:
                if session is None:
                    return False
                
                # Use PostgreSQL UPSERT
                stmt = insert(VendorRecord.__table__).values(vendors_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['vendor_code'],
                    set_=dict(
                        city_id=stmt.excluded.city_id,
                        latitude=stmt.excluded.latitude,
                        longitude=stmt.excluded.longitude,
                        status_id=stmt.excluded.status_id,
                        visible=stmt.excluded.visible,
                        open=stmt.excluded.open,
                        radius=stmt.excluded.radius,
                        vendor_name=stmt.excluded.vendor_name,
                        grade=stmt.excluded.grade,
                        business_line=stmt.excluded.business_line,
                        original_radius=stmt.excluded.original_radius,
                        updated_at=datetime.utcnow()
                    )
                )
                
                session.execute(stmt)
                
            # Update freshness tracking
            self._update_data_freshness('vendors', len(vendors_data))
            
            logger.info(f"Bulk inserted {len(vendors_data)} vendor records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to bulk insert vendors: {e}")
            return False
    
    def get_orders_filtered(self, 
                           city_id: Optional[int] = None,
                           business_line: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           limit: Optional[int] = None,
                           offset: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Get filtered orders data with pagination"""
        if not self._connected:
            return None
        
        try:
            query = "SELECT * FROM orders WHERE 1=1"
            params = {}
            
            if city_id is not None:
                query += " AND city_id = :city_id"
                params['city_id'] = city_id
            
            if business_line:
                query += " AND business_line = :business_line"
                params['business_line'] = business_line
            
            if start_date:
                query += " AND created_at >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND created_at <= :end_date"
                params['end_date'] = end_date
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += " LIMIT :limit"
                params['limit'] = limit
            
            if offset:
                query += " OFFSET :offset"
                params['offset'] = offset
            
            # Execute query and return DataFrame
            df = pd.read_sql(query, self.engine, params=params)
            return df
            
        except Exception as e:
            logger.error(f"Failed to get filtered orders: {e}")
            return None
    
    def get_vendors_filtered(self,
                           city_id: Optional[int] = None,
                           status_ids: Optional[List[int]] = None,
                           grades: Optional[List[str]] = None,
                           business_line: Optional[str] = None,
                           limit: Optional[int] = None,
                           offset: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Get filtered vendors data with pagination"""
        if not self._connected:
            return None
        
        try:
            query = "SELECT * FROM vendors WHERE 1=1"
            params = {}
            
            if city_id is not None:
                query += " AND city_id = :city_id"
                params['city_id'] = city_id
            
            if business_line:
                query += " AND business_line = :business_line"
                params['business_line'] = business_line
            
            if status_ids:
                placeholders = ','.join([f":status_{i}" for i in range(len(status_ids))])
                query += f" AND status_id IN ({placeholders})"
                for i, status_id in enumerate(status_ids):
                    params[f'status_{i}'] = status_id
            
            if grades:
                placeholders = ','.join([f":grade_{i}" for i in range(len(grades))])
                query += f" AND grade IN ({placeholders})"
                for i, grade in enumerate(grades):
                    params[f'grade_{i}'] = grade
            
            query += " ORDER BY vendor_code"
            
            if limit:
                query += " LIMIT :limit"
                params['limit'] = limit
            
            if offset:
                query += " OFFSET :offset"
                params['offset'] = offset
            
            # Execute query and return DataFrame
            df = pd.read_sql(query, self.engine, params=params)
            return df
            
        except Exception as e:
            logger.error(f"Failed to get filtered vendors: {e}")
            return None
    
    def get_aggregated_data(self, 
                           aggregation_type: str = 'hourly',
                           city_id: Optional[int] = None,
                           days_back: int = 7) -> Optional[pd.DataFrame]:
        """Get pre-aggregated data for faster queries"""
        if not self._connected:
            return None
        
        try:
            time_format = {
                'hourly': "date_trunc('hour', created_at)",
                'daily': "date_trunc('day', created_at)", 
                'weekly': "date_trunc('week', created_at)"
            }.get(aggregation_type, "date_trunc('hour', created_at)")
            
            query = f"""
            SELECT 
                {time_format} as time_bucket,
                city_id,
                business_line,
                COUNT(*) as order_count,
                COUNT(DISTINCT vendor_code) as vendor_count,
                AVG(latitude) as avg_lat,
                AVG(longitude) as avg_lng
            FROM orders 
            WHERE created_at >= NOW() - INTERVAL '{days_back} days'
            """
            
            params = {}
            if city_id is not None:
                query += " AND city_id = :city_id"
                params['city_id'] = city_id
            
            query += " GROUP BY time_bucket, city_id, business_line ORDER BY time_bucket DESC"
            
            df = pd.read_sql(query, self.engine, params=params)
            return df
            
        except Exception as e:
            logger.error(f"Failed to get aggregated data: {e}")
            return None
    
    def _update_data_freshness(self, data_type: str, record_count: int):
        """Update data freshness tracking"""
        try:
            with self.get_session() as session:
                if session is None:
                    return
                
                # Upsert freshness record
                stmt = insert(DataFreshness.__table__).values(
                    data_type=data_type,
                    last_updated=datetime.utcnow(),
                    record_count=record_count,
                    status='success'
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=['data_type'],
                    set_=dict(
                        last_updated=stmt.excluded.last_updated,
                        record_count=stmt.excluded.record_count,
                        status=stmt.excluded.status,
                        error_message=None
                    )
                )
                
                session.execute(stmt)
                
        except Exception as e:
            logger.error(f"Failed to update data freshness: {e}")
    
    def get_database_stats(self) -> DatabaseStats:
        """Get comprehensive database statistics"""
        if not self._connected:
            return DatabaseStats(connected=False)
        
        try:
            stats = DatabaseStats(connected=True)
            
            with self.get_session() as session:
                if session is None:
                    return DatabaseStats(connected=False)
                
                # Get record counts
                stats.total_orders = session.query(OrderRecord).count()
                stats.total_vendors = session.query(VendorRecord).count()
                
                # Get freshness data
                freshness_data = session.query(DataFreshness).all()
                for freshness in freshness_data:
                    if freshness.data_type == 'orders':
                        stats.last_order_update = freshness.last_updated
                    elif freshness.data_type == 'vendors':
                        stats.last_vendor_update = freshness.last_updated
                
                # Get connection pool stats
                if hasattr(self.engine.pool, 'size'):
                    stats.connection_pool_size = self.engine.pool.size()
                    stats.active_connections = self.engine.pool.checkedout()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return DatabaseStats(connected=False)
    
    def health_check(self) -> bool:
        """Check database health"""
        if not self._connected:
            return False
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")

# Database migration utilities
class DatabaseMigrator:
    """Handle database migrations and schema updates"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def migrate_to_latest(self) -> bool:
        """Run all pending migrations"""
        try:
            # Create base tables
            if not self.db_manager.create_tables():
                return False
            
            # Run specific migrations
            migrations = [
                self._create_indexes,
                self._create_views,
                self._optimize_tables
            ]
            
            for migration in migrations:
                if not migration():
                    logger.error(f"Migration failed: {migration.__name__}")
                    return False
            
            logger.info("All migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def _create_indexes(self) -> bool:
        """Create optimized indexes"""
        try:
            indexes = [
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_city_date ON orders(city_id, created_at)",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_vendor_date ON orders(vendor_code, created_at)",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_business_line ON orders(business_line) WHERE business_line IS NOT NULL",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_location ON orders USING GIST (ll_to_earth(latitude, longitude))",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vendors_city ON vendors(city_id)",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vendors_status ON vendors(status_id) WHERE status_id IS NOT NULL",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vendors_location ON vendors USING GIST (ll_to_earth(latitude, longitude))"
            ]
            
            with self.db_manager.engine.connect() as conn:
                for index_sql in indexes:
                    try:
                        conn.execute(text(index_sql))
                        conn.commit()
                    except Exception as e:
                        if "already exists" not in str(e):
                            logger.warning(f"Index creation warning: {e}")
            
            logger.info("Database indexes created/verified")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return False
    
    def _create_views(self) -> bool:
        """Create materialized views for common queries"""
        try:
            views = [
                """
                CREATE MATERIALIZED VIEW IF NOT EXISTS vendor_stats AS
                SELECT 
                    city_id,
                    business_line,
                    COUNT(*) as vendor_count,
                    COUNT(*) FILTER (WHERE visible > 0) as visible_count,
                    COUNT(*) FILTER (WHERE open > 0) as open_count,
                    AVG(radius) as avg_radius
                FROM vendors 
                WHERE vendor_code IS NOT NULL
                GROUP BY city_id, business_line
                """,
                """
                CREATE MATERIALIZED VIEW IF NOT EXISTS daily_order_stats AS
                SELECT 
                    DATE(created_at) as order_date,
                    city_id,
                    business_line,
                    COUNT(*) as order_count,
                    COUNT(DISTINCT vendor_code) as unique_vendors
                FROM orders
                WHERE created_at >= NOW() - INTERVAL '30 days'
                GROUP BY DATE(created_at), city_id, business_line
                """
            ]
            
            with self.db_manager.engine.connect() as conn:
                for view_sql in views:
                    try:
                        conn.execute(text(view_sql))
                        conn.commit()
                    except Exception as e:
                        if "already exists" not in str(e):
                            logger.warning(f"View creation warning: {e}")
            
            logger.info("Database views created/verified")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create views: {e}")
            return False
    
    def _optimize_tables(self) -> bool:
        """Optimize table settings"""
        try:
            optimizations = [
                "ANALYZE orders",
                "ANALYZE vendors", 
                "REFRESH MATERIALIZED VIEW CONCURRENTLY vendor_stats",
                "REFRESH MATERIALIZED VIEW CONCURRENTLY daily_order_stats"
            ]
            
            with self.db_manager.engine.connect() as conn:
                for opt_sql in optimizations:
                    try:
                        conn.execute(text(opt_sql))
                        conn.commit()
                    except Exception as e:
                        logger.warning(f"Optimization warning: {e}")
            
            logger.info("Database optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize tables: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager()
db_migrator = DatabaseMigrator(db_manager)

def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    return db_manager

def get_db_migrator() -> DatabaseMigrator:
    """Get the global database migrator instance"""
    return db_migrator