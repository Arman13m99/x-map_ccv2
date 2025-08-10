# services/map_service.py - Heatmap and coverage calculations
import logging
import time
import asyncio
from typing import Optional, List, Dict, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import transform
import functools

from config import get_config
from cache import get_cache_manager
from models import HeatmapGenerator, CoverageCalculator, DataProcessor
from services.data_service import get_data_service

logger = logging.getLogger(__name__)

class HeatmapType(Enum):
    """Heatmap visualization types"""
    ORDER_DENSITY = "order_density"
    VENDOR_DENSITY = "vendor_density"
    USER_DENSITY = "user_density"
    REVENUE_DENSITY = "revenue_density"
    COVERAGE_QUALITY = "coverage_quality"

class AggregationMethod(Enum):
    """Data aggregation methods"""
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    PERCENTILE = "percentile"

@dataclass
class HeatmapRequest:
    """Heatmap generation request parameters"""
    heatmap_type: HeatmapType
    city_id: int
    business_line: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    zoom_level: int = 11
    aggregation_method: AggregationMethod = AggregationMethod.COUNT
    include_coverage: bool = False
    max_points: int = 10000

@dataclass
class HeatmapResult:
    """Heatmap generation result"""
    success: bool
    points: List[Dict[str, float]]
    metadata: Dict[str, Any]
    generation_time: float
    cache_key: Optional[str] = None
    error: Optional[str] = None

@dataclass
class CoverageAnalysis:
    """Coverage analysis result"""
    total_area: float
    covered_area: float
    coverage_percentage: float
    uncovered_regions: List[Dict[str, Any]]
    vendor_distribution: Dict[str, int]
    quality_metrics: Dict[str, float]

class MapService:
    """Advanced map visualization service with optimized heatmap and coverage calculations"""
    
    def __init__(self):
        self.config = get_config()
        self.cache_manager = get_cache_manager()
        self.data_service = get_data_service()
        self.heatmap_generator = HeatmapGenerator()
        self.coverage_calculator = CoverageCalculator()
        self.data_processor = DataProcessor()
        
        # Performance tracking
        self.generation_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'generation_time': 0.0,
            'average_points_per_heatmap': 0
        }
    
    async def generate_heatmap(self, request: HeatmapRequest) -> HeatmapResult:
        """Generate optimized heatmap with caching and background processing"""
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_heatmap_cache_key(request)
            
            # Try cache first
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.generation_stats['cache_hits'] += 1
                self.generation_stats['total_requests'] += 1
                
                return HeatmapResult(
                    success=True,
                    points=cached_result['points'],
                    metadata=cached_result['metadata'],
                    generation_time=time.time() - start_time,
                    cache_key=cache_key
                )
            
            # Generate new heatmap
            result = await self._generate_heatmap_internal(request)
            
            # Cache successful results
            if result.success and len(result.points) > 0:
                cache_data = {
                    'points': result.points,
                    'metadata': result.metadata
                }
                # Cache for different durations based on data freshness requirements
                cache_ttl = self._get_cache_ttl(request.heatmap_type)
                self.cache_manager.set(cache_key, cache_data, ttl=cache_ttl)
                result.cache_key = cache_key
            
            # Update stats
            self.generation_stats['total_requests'] += 1
            self.generation_stats['generation_time'] += result.generation_time
            if result.success:
                self.generation_stats['average_points_per_heatmap'] = (
                    (self.generation_stats['average_points_per_heatmap'] * (self.generation_stats['total_requests'] - 1) + len(result.points))
                    / self.generation_stats['total_requests']
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
            return HeatmapResult(
                success=False,
                points=[],
                metadata={},
                generation_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _generate_heatmap_internal(self, request: HeatmapRequest) -> HeatmapResult:
        """Internal heatmap generation logic"""
        start_time = time.time()
        
        try:
            # Get appropriate data based on heatmap type
            if request.heatmap_type in [HeatmapType.ORDER_DENSITY, HeatmapType.USER_DENSITY, HeatmapType.REVENUE_DENSITY]:
                data_result = await self.data_service.get_orders_data(
                    city_id=request.city_id,
                    business_line=request.business_line,
                    start_date=request.start_date,
                    end_date=request.end_date
                )
                
                if not data_result.success or data_result.data is None:
                    return HeatmapResult(
                        success=False,
                        points=[],
                        metadata={},
                        generation_time=time.time() - start_time,
                        error="Failed to fetch orders data"
                    )
                
                df = data_result.data
                
            elif request.heatmap_type == HeatmapType.VENDOR_DENSITY:
                data_result = await self.data_service.get_vendors_data(
                    city_id=request.city_id,
                    business_line=request.business_line
                )
                
                if not data_result.success or data_result.data is None:
                    return HeatmapResult(
                        success=False,
                        points=[],
                        metadata={},
                        generation_time=time.time() - start_time,
                        error="Failed to fetch vendors data"
                    )
                
                df = data_result.data
                
            else:
                return HeatmapResult(
                    success=False,
                    points=[],
                    metadata={},
                    generation_time=time.time() - start_time,
                    error=f"Unsupported heatmap type: {request.heatmap_type}"
                )
            
            # Generate heatmap points using optimized method
            if request.heatmap_type == HeatmapType.ORDER_DENSITY:
                points = await self._generate_order_density_heatmap(df, request)
            elif request.heatmap_type == HeatmapType.VENDOR_DENSITY:
                points = await self._generate_vendor_density_heatmap(df, request)
            elif request.heatmap_type == HeatmapType.USER_DENSITY:
                points = await self._generate_user_density_heatmap(df, request)
            elif request.heatmap_type == HeatmapType.REVENUE_DENSITY:
                points = await self._generate_revenue_density_heatmap(df, request)
            else:
                points = []
            
            # Apply point limit
            if len(points) > request.max_points:
                points = self._subsample_points(points, request.max_points)
            
            # Generate metadata
            metadata = self._generate_metadata(df, points, request)
            
            return HeatmapResult(
                success=True,
                points=points,
                metadata=metadata,
                generation_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Internal heatmap generation failed: {e}")
            return HeatmapResult(
                success=False,
                points=[],
                metadata={},
                generation_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _generate_order_density_heatmap(self, df: pd.DataFrame, request: HeatmapRequest) -> List[Dict[str, float]]:
        """Generate order density heatmap"""
        try:
            # Use asyncio to run CPU-intensive work in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=2) as executor:
                future = loop.run_in_executor(
                    executor,
                    self.heatmap_generator.generate_heatmap_data,
                    df,
                    'latitude',
                    'longitude',
                    None,
                    request.aggregation_method.value
                )
                points = await future
            
            return points
            
        except Exception as e:
            logger.error(f"Order density heatmap generation failed: {e}")
            return []
    
    async def _generate_vendor_density_heatmap(self, df: pd.DataFrame, request: HeatmapRequest) -> List[Dict[str, float]]:
        """Generate vendor density heatmap"""
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=2) as executor:
                future = loop.run_in_executor(
                    executor,
                    self.heatmap_generator.generate_heatmap_data,
                    df,
                    'latitude',
                    'longitude',
                    'radius',  # Use radius as weight
                    request.aggregation_method.value
                )
                points = await future
            
            return points
            
        except Exception as e:
            logger.error(f"Vendor density heatmap generation failed: {e}")
            return []
    
    async def _generate_user_density_heatmap(self, df: pd.DataFrame, request: HeatmapRequest) -> List[Dict[str, float]]:
        """Generate user density heatmap (orders grouped by unique users)"""
        try:
            # Group by approximate user location (rounded coordinates)
            df_users = df.copy()
            df_users['lat_rounded'] = df_users['latitude'].round(4)
            df_users['lng_rounded'] = df_users['longitude'].round(4)
            
            # Aggregate by user location
            df_user_agg = df_users.groupby(['lat_rounded', 'lng_rounded']).agg({
                'vendor_code': 'count'  # Count orders per location
            }).reset_index()
            
            df_user_agg.rename(columns={
                'lat_rounded': 'latitude',
                'lng_rounded': 'longitude',
                'vendor_code': 'order_count'
            }, inplace=True)
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=2) as executor:
                future = loop.run_in_executor(
                    executor,
                    self.heatmap_generator.generate_heatmap_data,
                    df_user_agg,
                    'latitude',
                    'longitude',
                    'order_count',
                    request.aggregation_method.value
                )
                points = await future
            
            return points
            
        except Exception as e:
            logger.error(f"User density heatmap generation failed: {e}")
            return []
    
    async def _generate_revenue_density_heatmap(self, df: pd.DataFrame, request: HeatmapRequest) -> List[Dict[str, float]]:
        """Generate revenue density heatmap"""
        try:
            # Simulate revenue data (in real implementation, this would come from actual revenue data)
            df_revenue = df.copy()
            
            # Add simulated revenue based on business line and organic status
            df_revenue['estimated_revenue'] = np.random.uniform(10, 100, len(df_revenue))
            
            # Adjust revenue by business line
            if 'business_line' in df_revenue.columns:
                bl_multipliers = {
                    'food': 1.0,
                    'grocery': 0.8,
                    'pharmacy': 1.2,
                    'flowers': 0.6
                }
                df_revenue['bl_multiplier'] = df_revenue['business_line'].map(bl_multipliers).fillna(1.0)
                df_revenue['estimated_revenue'] *= df_revenue['bl_multiplier']
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=2) as executor:
                future = loop.run_in_executor(
                    executor,
                    self.heatmap_generator.generate_heatmap_data,
                    df_revenue,
                    'latitude',
                    'longitude',
                    'estimated_revenue',
                    request.aggregation_method.value
                )
                points = await future
            
            return points
            
        except Exception as e:
            logger.error(f"Revenue density heatmap generation failed: {e}")
            return []
    
    async def analyze_coverage(self, 
                              city_id: int,
                              business_line: Optional[str] = None,
                              include_quality_metrics: bool = True) -> CoverageAnalysis:
        """Analyze vendor coverage for a city/business line"""
        try:
            # Get vendors data
            vendors_result = await self.data_service.get_vendors_data(
                city_id=city_id,
                business_line=business_line
            )
            
            if not vendors_result.success or vendors_result.data is None:
                return CoverageAnalysis(
                    total_area=0,
                    covered_area=0,
                    coverage_percentage=0,
                    uncovered_regions=[],
                    vendor_distribution={},
                    quality_metrics={}
                )
            
            vendors_df = vendors_result.data
            
            # Convert to GeoDataFrame
            geometry = [Point(row['longitude'], row['latitude']) 
                       for _, row in vendors_df.iterrows()]
            vendors_gdf = gpd.GeoDataFrame(vendors_df, geometry=geometry)
            
            # Calculate coverage using background processing
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(
                    executor,
                    self._calculate_coverage_internal,
                    vendors_gdf
                )
                coverage_data = await future
            
            # Analyze vendor distribution
            distribution = self._analyze_vendor_distribution(vendors_df)
            
            # Calculate quality metrics if requested
            quality_metrics = {}
            if include_quality_metrics:
                quality_metrics = self._calculate_coverage_quality_metrics(
                    vendors_gdf, coverage_data
                )
            
            return CoverageAnalysis(
                total_area=coverage_data.get('total_area', 0),
                covered_area=coverage_data.get('covered_area', 0),
                coverage_percentage=coverage_data.get('coverage_percentage', 0),
                uncovered_regions=coverage_data.get('uncovered_regions', []),
                vendor_distribution=distribution,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            return CoverageAnalysis(
                total_area=0,
                covered_area=0,
                coverage_percentage=0,
                uncovered_regions=[],
                vendor_distribution={},
                quality_metrics={}
            )
    
    def _calculate_coverage_internal(self, vendors_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Calculate coverage metrics (CPU-intensive, runs in thread pool)"""
        try:
            # Get city bounds
            bounds = vendors_gdf.total_bounds
            
            # Calculate grid coverage
            coverage_grid = self.coverage_calculator.calculate_grid_coverage(
                vendors_gdf,
                grid_size=0.01,  # ~1km grid
                bounds=bounds
            )
            
            # Calculate metrics
            total_grid_cells = len(coverage_grid) if coverage_grid else 0
            covered_cells = len([cell for cell in coverage_grid if cell.get('coverage', 0) > 0])
            
            coverage_percentage = (covered_cells / total_grid_cells * 100) if total_grid_cells > 0 else 0
            
            # Find uncovered regions (simplified)
            uncovered_regions = [
                {
                    'lat': cell['lat'],
                    'lng': cell['lng'],
                    'severity': 'high' if cell.get('coverage', 0) == 0 else 'medium'
                }
                for cell in coverage_grid
                if cell.get('coverage', 0) < 25  # Less than 25% coverage
            ][:100]  # Limit to top 100 uncovered areas
            
            return {
                'total_area': total_grid_cells,
                'covered_area': covered_cells,
                'coverage_percentage': coverage_percentage,
                'uncovered_regions': uncovered_regions
            }
            
        except Exception as e:
            logger.error(f"Internal coverage calculation failed: {e}")
            return {
                'total_area': 0,
                'covered_area': 0,
                'coverage_percentage': 0,
                'uncovered_regions': []
            }
    
    def _analyze_vendor_distribution(self, vendors_df: pd.DataFrame) -> Dict[str, int]:
        """Analyze vendor distribution by various attributes"""
        try:
            distribution = {}
            
            # By status
            if 'status_id' in vendors_df.columns:
                status_dist = vendors_df['status_id'].value_counts().to_dict()
                distribution['by_status'] = {str(k): v for k, v in status_dist.items()}
            
            # By grade
            if 'grade' in vendors_df.columns:
                grade_dist = vendors_df['grade'].value_counts().to_dict()
                distribution['by_grade'] = grade_dist
            
            # By business line
            if 'business_line' in vendors_df.columns:
                bl_dist = vendors_df['business_line'].value_counts().to_dict()
                distribution['by_business_line'] = bl_dist
            
            return distribution
            
        except Exception as e:
            logger.error(f"Vendor distribution analysis failed: {e}")
            return {}
    
    def _calculate_coverage_quality_metrics(self, 
                                          vendors_gdf: gpd.GeoDataFrame,
                                          coverage_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate coverage quality metrics"""
        try:
            metrics = {}
            
            # Vendor density
            total_area_km2 = coverage_data.get('total_area', 1) * 1.21  # ~1km² per grid cell
            metrics['vendor_density_per_km2'] = len(vendors_gdf) / total_area_km2
            
            # Coverage efficiency
            metrics['coverage_efficiency'] = coverage_data.get('coverage_percentage', 0)
            
            # Average service radius
            if 'radius' in vendors_gdf.columns:
                metrics['average_service_radius'] = vendors_gdf['radius'].mean()
                metrics['service_radius_std'] = vendors_gdf['radius'].std()
            
            # Distribution evenness (how evenly distributed vendors are)
            if len(vendors_gdf) > 0:
                # Calculate spatial clustering using nearest neighbor distances
                coords = np.array([[p.x, p.y] for p in vendors_gdf.geometry])
                if len(coords) > 1:
                    from scipy.spatial.distance import cdist
                    distances = cdist(coords, coords)
                    np.fill_diagonal(distances, np.inf)
                    nearest_distances = np.min(distances, axis=1)
                    metrics['average_nearest_neighbor_distance'] = np.mean(nearest_distances)
                    metrics['distribution_evenness'] = 1 / (1 + np.std(nearest_distances))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Coverage quality metrics calculation failed: {e}")
            return {}
    
    def _subsample_points(self, points: List[Dict[str, float]], max_points: int) -> List[Dict[str, float]]:
        """Subsample points while preserving spatial distribution"""
        if len(points) <= max_points:
            return points
        
        try:
            # Sort by intensity (keep high-intensity points)
            sorted_points = sorted(points, key=lambda p: p.get('intensity', 0), reverse=True)
            
            # Take top points by intensity
            high_intensity_count = min(max_points // 2, len(sorted_points))
            high_intensity_points = sorted_points[:high_intensity_count]
            
            # Sample remaining points spatially
            remaining_points = sorted_points[high_intensity_count:]
            remaining_count = max_points - high_intensity_count
            
            if len(remaining_points) > remaining_count:
                # Simple spatial sampling (could be improved with k-means)
                step = len(remaining_points) // remaining_count
                sampled_remaining = remaining_points[::step][:remaining_count]
            else:
                sampled_remaining = remaining_points
            
            return high_intensity_points + sampled_remaining
            
        except Exception as e:
            logger.error(f"Point subsampling failed: {e}")
            return points[:max_points]  # Fallback to simple truncation
    
    def _generate_metadata(self, 
                          df: pd.DataFrame, 
                          points: List[Dict[str, float]], 
                          request: HeatmapRequest) -> Dict[str, Any]:
        """Generate heatmap metadata"""
        try:
            metadata = {
                'heatmap_type': request.heatmap_type.value,
                'city_id': request.city_id,
                'business_line': request.business_line,
                'zoom_level': request.zoom_level,
                'aggregation_method': request.aggregation_method.value,
                'total_data_points': len(df),
                'heatmap_points': len(points),
                'date_range': {
                    'start': request.start_date,
                    'end': request.end_date
                }
            }
            
            # Add intensity statistics
            if points:
                intensities = [p.get('intensity', 0) for p in points]
                metadata['intensity_stats'] = {
                    'min': min(intensities),
                    'max': max(intensities),
                    'mean': np.mean(intensities),
                    'std': np.std(intensities)
                }
            
            # Add geographical bounds
            if 'latitude' in df.columns and 'longitude' in df.columns:
                metadata['bounds'] = {
                    'north': df['latitude'].max(),
                    'south': df['latitude'].min(),
                    'east': df['longitude'].max(),
                    'west': df['longitude'].min()
                }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_heatmap_cache_key(self, request: HeatmapRequest) -> str:
        """Generate cache key for heatmap request"""
        key_parts = [
            'heatmap',
            request.heatmap_type.value,
            str(request.city_id),
            request.business_line or 'all',
            request.start_date or '',
            request.end_date or '',
            str(request.zoom_level),
            request.aggregation_method.value,
            str(request.max_points)
        ]
        
        return ':'.join(key_parts)
    
    def _get_cache_ttl(self, heatmap_type: HeatmapType) -> int:
        """Get cache TTL based on heatmap type"""
        ttl_map = {
            HeatmapType.ORDER_DENSITY: 1800,  # 30 minutes
            HeatmapType.USER_DENSITY: 3600,   # 1 hour
            HeatmapType.VENDOR_DENSITY: 600,  # 10 minutes
            HeatmapType.REVENUE_DENSITY: 1800, # 30 minutes
            HeatmapType.COVERAGE_QUALITY: 7200 # 2 hours
        }
        
        return ttl_map.get(heatmap_type, 1800)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get map service performance statistics"""
        total_requests = self.generation_stats['total_requests']
        cache_hit_rate = (
            self.generation_stats['cache_hits'] / total_requests * 100
            if total_requests > 0 else 0
        )
        
        average_generation_time = (
            self.generation_stats['generation_time'] / (total_requests - self.generation_stats['cache_hits'])
            if (total_requests - self.generation_stats['cache_hits']) > 0 else 0
        )
        
        return {
            'total_heatmap_requests': total_requests,
            'cache_hit_rate': cache_hit_rate,
            'average_generation_time': average_generation_time,
            'average_points_per_heatmap': self.generation_stats['average_points_per_heatmap']
        }
    
    async def batch_generate_heatmaps(self, requests: List[HeatmapRequest]) -> List[HeatmapResult]:
        """Generate multiple heatmaps concurrently"""
        try:
            tasks = [self.generate_heatmap(request) for request in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            final_results = []
            for result in results:
                if isinstance(result, Exception):
                    final_results.append(HeatmapResult(
                        success=False,
                        points=[],
                        metadata={},
                        generation_time=0.0,
                        error=str(result)
                    ))
                else:
                    final_results.append(result)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Batch heatmap generation failed: {e}")
            return [HeatmapResult(
                success=False,
                points=[],
                metadata={},
                generation_time=0.0,
                error=str(e)
            ) for _ in requests]

# Global map service instance
map_service = MapService()

def get_map_service() -> MapService:
    """Get the global map service instance"""
    return map_service