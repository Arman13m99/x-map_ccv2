# models.py - Data models for vendors, orders, polygons
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

@dataclass
class OrderData:
    """Order data model with validation and processing"""
    city_id: int
    vendor_code: str
    business_line: str
    marketing_area: Optional[str]
    latitude: float
    longitude: float
    created_at: datetime
    organic: int = 0
    
    def __post_init__(self):
        """Validate order data"""
        if not isinstance(self.latitude, (int, float)) or not (-90 <= self.latitude <= 90):
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not isinstance(self.longitude, (int, float)) or not (-180 <= self.longitude <= 180):
            raise ValueError(f"Invalid longitude: {self.longitude}")
        if self.organic not in [0, 1]:
            raise ValueError(f"Organic must be 0 or 1, got: {self.organic}")

@dataclass
class VendorData:
    """Vendor data model with validation and processing"""
    vendor_code: str
    city_id: int
    latitude: float
    longitude: float
    status_id: float
    visible: float
    open: float
    radius: float
    vendor_name: Optional[str] = None
    grade: str = 'Ungraded'
    business_line: Optional[str] = None
    original_radius: Optional[float] = None
    
    def __post_init__(self):
        """Validate vendor data"""
        if not isinstance(self.latitude, (int, float)) or not (-90 <= self.latitude <= 90):
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not isinstance(self.longitude, (int, float)) or not (-180 <= self.longitude <= 180):
            raise ValueError(f"Invalid longitude: {self.longitude}")
        if self.original_radius is None:
            self.original_radius = self.radius
    
    @property
    def geometry(self) -> Point:
        """Get vendor location as Shapely Point"""
        return Point(self.longitude, self.latitude)

@dataclass
class HeatmapPoint:
    """Heatmap point data model"""
    latitude: float
    longitude: float
    intensity: float
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization"""
        return {
            'lat': self.latitude,
            'lng': self.longitude,
            'intensity': self.intensity,
            'weight': self.weight
        }

class DataProcessor:
    """Data processing utilities for optimization and validation"""
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        if df.empty:
            return df
        
        df_optimized = df.copy()
        
        # Optimize numeric columns
        for col in df_optimized.select_dtypes(include=[np.number]).columns:
            col_type = df_optimized[col].dtype
            
            if col_type != np.object:
                c_min = df_optimized[col].min()
                c_max = df_optimized[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df_optimized[col] = df_optimized[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df_optimized[col] = df_optimized[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df_optimized[col] = df_optimized[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df_optimized[col] = df_optimized[col].astype(np.float32)
        
        # Convert string columns to categories if they have low cardinality
        for col in df_optimized.select_dtypes(include=['object']).columns:
            if col in df_optimized.columns:
                num_unique_values = len(df_optimized[col].unique())
                num_total_values = len(df_optimized[col])
                
                if num_unique_values / num_total_values < 0.5:  # Less than 50% unique
                    df_optimized[col] = df_optimized[col].astype('category')
        
        return df_optimized
    
    @staticmethod
    def validate_coordinates(df: pd.DataFrame, lat_col: str = 'latitude', lng_col: str = 'longitude') -> pd.DataFrame:
        """Validate and clean coordinate data"""
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # Check if columns exist
        if lat_col not in df_clean.columns or lng_col not in df_clean.columns:
            logger.warning(f"Coordinate columns {lat_col}, {lng_col} not found")
            return df_clean
        
        # Remove invalid coordinates
        original_count = len(df_clean)
        
        df_clean = df_clean[
            (df_clean[lat_col].between(-90, 90)) &
            (df_clean[lng_col].between(-180, 180)) &
            (df_clean[lat_col].notna()) &
            (df_clean[lng_col].notna())
        ]
        
        removed_count = original_count - len(df_clean)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with invalid coordinates")
        
        return df_clean
    
    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 100000):
        """Generator to process DataFrame in chunks"""
        if df.empty:
            return
        
        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            yield df.iloc[start:end]
    
    @staticmethod
    def create_spatial_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Create spatial index for faster spatial operations"""
        if gdf.empty:
            return gdf
        
        try:
            # Create spatial index
            gdf_indexed = gdf.copy()
            gdf_indexed.sindex  # This creates the spatial index
            logger.info(f"Spatial index created for {len(gdf_indexed)} geometries")
            return gdf_indexed
        except Exception as e:
            logger.warning(f"Failed to create spatial index: {e}")
            return gdf

class HeatmapGenerator:
    """Optimized heatmap generation with caching and chunking"""
    
    def __init__(self, chunk_size: int = 100000):
        self.chunk_size = chunk_size
    
    def generate_heatmap_data(self, 
                            df: pd.DataFrame,
                            lat_col: str = 'latitude',
                            lng_col: str = 'longitude',
                            weight_col: str = None,
                            intensity_method: str = 'count') -> List[Dict[str, float]]:
        """
        Generate optimized heatmap data
        
        Args:
            df: Input DataFrame
            lat_col: Latitude column name
            lng_col: Longitude column name
            weight_col: Weight column name (optional)
            intensity_method: 'count', 'sum', or 'mean'
        """
        if df.empty:
            return []
        
        try:
            # Validate coordinates
            df_clean = DataProcessor.validate_coordinates(df, lat_col, lng_col)
            if df_clean.empty:
                logger.warning("No valid coordinates found for heatmap")
                return []
            
            # Aggregate points by coordinates (round to reasonable precision)
            precision = 4  # ~11m precision
            df_clean[f'{lat_col}_rounded'] = df_clean[lat_col].round(precision)
            df_clean[f'{lng_col}_rounded'] = df_clean[lng_col].round(precision)
            
            # Group by rounded coordinates
            group_cols = [f'{lat_col}_rounded', f'{lng_col}_rounded']
            
            if weight_col and weight_col in df_clean.columns:
                if intensity_method == 'sum':
                    agg_dict = {weight_col: 'sum'}
                elif intensity_method == 'mean':
                    agg_dict = {weight_col: 'mean'}
                else:
                    agg_dict = {weight_col: 'count'}
            else:
                # Count method
                agg_dict = {lat_col: 'count'}
                weight_col = lat_col
            
            # Perform aggregation
            aggregated = df_clean.groupby(group_cols).agg(agg_dict).reset_index()
            
            # Rename columns
            aggregated.rename(columns={
                f'{lat_col}_rounded': 'lat',
                f'{lng_col}_rounded': 'lng',
                weight_col: 'intensity'
            }, inplace=True)
            
            # Normalize intensity (0-100 scale)
            if len(aggregated) > 0:
                min_intensity = aggregated['intensity'].min()
                max_intensity = aggregated['intensity'].max()
                
                if max_intensity > min_intensity:
                    aggregated['intensity'] = (
                        (aggregated['intensity'] - min_intensity) / 
                        (max_intensity - min_intensity) * 100
                    )
                else:
                    aggregated['intensity'] = 50  # Default middle value
            
            # Convert to list of dictionaries
            heatmap_data = aggregated[['lat', 'lng', 'intensity']].to_dict('records')
            
            logger.info(f"Generated heatmap with {len(heatmap_data)} points from {len(df_clean)} records")
            return heatmap_data
            
        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            return []
    
    def generate_density_heatmap(self,
                               df: pd.DataFrame,
                               lat_col: str = 'latitude',
                               lng_col: str = 'longitude',
                               radius_km: float = 1.0) -> List[Dict[str, float]]:
        """Generate density-based heatmap using kernel density estimation"""
        try:
            from sklearn.neighbors import KernelDensity
            from sklearn.model_selection import GridSearchCV
            
            df_clean = DataProcessor.validate_coordinates(df, lat_col, lng_col)
            if len(df_clean) < 10:
                logger.warning("Not enough points for density heatmap")
                return self.generate_heatmap_data(df_clean, lat_col, lng_col)
            
            # Extract coordinates
            coords = df_clean[[lat_col, lng_col]].values
            
            # Use cross-validation to find optimal bandwidth
            bandwidths = np.logspace(-3, 0, 20)
            grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                              {'bandwidth': bandwidths},
                              cv=min(5, len(coords) // 10))
            grid.fit(coords)
            
            kde = grid.best_estimator_
            
            # Create grid for density estimation
            lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
            lng_min, lng_max = coords[:, 1].min(), coords[:, 1].max()
            
            # Adjust grid resolution based on data density
            resolution = min(100, max(20, int(np.sqrt(len(coords)) / 2)))
            
            lat_grid = np.linspace(lat_min, lat_max, resolution)
            lng_grid = np.linspace(lng_min, lng_max, resolution)
            lat_mesh, lng_mesh = np.meshgrid(lat_grid, lng_grid)
            
            # Compute density
            grid_coords = np.vstack([lat_mesh.ravel(), lng_mesh.ravel()]).T
            log_density = kde.score_samples(grid_coords)
            density = np.exp(log_density)
            
            # Normalize density
            density_normalized = (density - density.min()) / (density.max() - density.min()) * 100
            
            # Convert to heatmap format
            heatmap_data = []
            for i, (lat, lng, intensity) in enumerate(zip(lat_mesh.ravel(), lng_mesh.ravel(), density_normalized)):
                if intensity > 5:  # Only include points with meaningful density
                    heatmap_data.append({
                        'lat': float(lat),
                        'lng': float(lng),
                        'intensity': float(intensity)
                    })
            
            logger.info(f"Generated density heatmap with {len(heatmap_data)} grid points")
            return heatmap_data
            
        except ImportError:
            logger.warning("sklearn not available, falling back to simple heatmap")
            return self.generate_heatmap_data(df, lat_col, lng_col)
        except Exception as e:
            logger.error(f"Error in density heatmap: {e}")
            return self.generate_heatmap_data(df, lat_col, lng_col)

class CoverageCalculator:
    """Calculate coverage metrics with optimization"""
    
    def __init__(self):
        pass
    
    def calculate_grid_coverage(self,
                              vendors_gdf: gpd.GeoDataFrame,
                              grid_size: float = 0.01,
                              bounds: Tuple[float, float, float, float] = None) -> List[Dict[str, Any]]:
        """
        Calculate coverage on a grid system
        
        Args:
            vendors_gdf: GeoDataFrame of vendors with geometry and radius
            grid_size: Grid cell size in degrees
            bounds: (min_lng, min_lat, max_lng, max_lat)
        """
        try:
            if vendors_gdf.empty or 'geometry' not in vendors_gdf.columns:
                logger.warning("No valid vendor geometry data for coverage calculation")
                return []
            
            # Get bounds
            if bounds is None:
                total_bounds = vendors_gdf.total_bounds
                min_lng, min_lat, max_lng, max_lat = total_bounds
            else:
                min_lng, min_lat, max_lng, max_lat = bounds
            
            # Create grid
            lng_range = np.arange(min_lng, max_lng + grid_size, grid_size)
            lat_range = np.arange(min_lat, max_lat + grid_size, grid_size)
            
            coverage_data = []
            
            # Process in chunks to manage memory
            for i, lat in enumerate(lat_range[:-1]):
                for j, lng in enumerate(lng_range[:-1]):
                    # Create grid cell
                    cell_polygon = Polygon([
                        (lng, lat),
                        (lng + grid_size, lat),
                        (lng + grid_size, lat + grid_size),
                        (lng, lat + grid_size),
                        (lng, lat)
                    ])
                    
                    # Check coverage
                    covered = False
                    covering_vendors = 0
                    
                    for _, vendor in vendors_gdf.iterrows():
                        if pd.isna(vendor.get('radius', np.nan)):
                            continue
                            
                        # Create vendor coverage circle
                        vendor_buffer = vendor['geometry'].buffer(vendor.get('radius', 0) / 111000)  # Convert meters to degrees
                        
                        if vendor_buffer.intersects(cell_polygon):
                            covered = True
                            covering_vendors += 1
                    
                    # Calculate coverage percentage
                    if covered:
                        coverage_percentage = min(100, covering_vendors * 25)  # Scale coverage
                    else:
                        coverage_percentage = 0
                    
                    if coverage_percentage > 0:  # Only include covered areas
                        coverage_data.append({
                            'lat': lat + grid_size / 2,
                            'lng': lng + grid_size / 2,
                            'coverage': coverage_percentage,
                            'vendor_count': covering_vendors
                        })
            
            logger.info(f"Calculated coverage for {len(coverage_data)} grid cells")
            return coverage_data
            
        except Exception as e:
            logger.error(f"Error calculating grid coverage: {e}")
            return []

# Utility functions for data processing
def filter_by_city(df: pd.DataFrame, city_id: int) -> pd.DataFrame:
    """Filter DataFrame by city ID"""
    if df.empty or 'city_id' not in df.columns:
        return df
    return df[df['city_id'] == city_id].copy()

def filter_by_business_line(df: pd.DataFrame, business_line: str) -> pd.DataFrame:
    """Filter DataFrame by business line"""
    if df.empty or 'business_line' not in df.columns or not business_line:
        return df
    return df[df['business_line'] == business_line].copy()

def filter_by_date_range(df: pd.DataFrame, 
                        start_date: str = None, 
                        end_date: str = None,
                        date_col: str = 'created_at') -> pd.DataFrame:
    """Filter DataFrame by date range"""
    if df.empty or date_col not in df.columns:
        return df
    
    df_filtered = df.copy()
    
    if start_date:
        start_date = pd.to_datetime(start_date)
        df_filtered = df_filtered[df_filtered[date_col] >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        df_filtered = df_filtered[df_filtered[date_col] <= end_date]
    
    return df_filtered

def get_memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """Get memory usage information for DataFrame"""
    if df.empty:
        return {'total': '0 MB', 'per_column': {}}
    
    memory_usage = df.memory_usage(deep=True)
    total_mb = memory_usage.sum() / (1024 * 1024)
    
    per_column = {}
    for col in df.columns:
        col_mb = memory_usage[col] / (1024 * 1024)
        per_column[col] = f"{col_mb:.2f} MB"
    
    return {
        'total': f"{total_mb:.2f} MB",
        'rows': len(df),
        'columns': len(df.columns),
        'per_column': per_column
    }