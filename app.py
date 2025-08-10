# app.py - Production-ready Tapsi Food Map Dashboard
import os
from datetime import datetime
import re
import threading
import webbrowser
import time
import random
import logging
import atexit
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, Polygon
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import hashlib
from functools import lru_cache
from scipy import stats

# Import new production components
from config import get_config
from scheduler import get_scheduler
from cache import get_cache_decorator, get_cache_manager, warm_cache
from models import DataProcessor, HeatmapGenerator, CoverageCalculator, filter_by_city, filter_by_business_line, filter_by_date_range

# --- Production Configuration ---
config = get_config()
data_scheduler = get_scheduler()
cache_decorator = get_cache_decorator()
cache_manager = get_cache_manager()

# Initialize data processors
data_processor = DataProcessor()
heatmap_generator = HeatmapGenerator(chunk_size=config.app.chunk_size)
coverage_calculator = CoverageCalculator()

# Setup logging
logger = logging.getLogger(__name__) 

# --- Initialize Flask App ---
app = Flask(__name__, static_folder=config.app.public_dir, static_url_path='')
CORS(app)

# Configure Flask from config
app.config['SECRET_KEY'] = config.app.secret_key
app.config['DEBUG'] = config.app.debug
app.config['TESTING'] = config.app.testing
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# --- Application State ---
# Data is now managed by the scheduler, no global variables needed
city_id_map = config.app.city_id_map

# Application startup flag
app_initialized = False
city_name_to_id_map = {v: k for k, v in city_id_map.items()}

# Simple cache for coverage calculations
coverage_cache = {}
CACHE_SIZE = 100  # Limit cache size to prevent memory issues

# City boundaries for grid generation (approximate)
city_boundaries = {
    "tehran": {"min_lat": 35.5, "max_lat": 35.85, "min_lng": 51.1, "max_lng": 51.7},
    "mashhad": {"min_lat": 36.15, "max_lat": 36.45, "min_lng": 59.35, "max_lng": 59.8},
    "shiraz": {"min_lat": 29.5, "max_lat": 29.75, "min_lng": 52.4, "max_lng": 52.7}
}

# --- Helper Functions ---
def safe_tolist(series):
    if series.empty:
        return []
    cleaned = series.dropna().unique()
    if pd.api.types.is_numeric_dtype(cleaned.dtype):
        return [item.item() if hasattr(item, 'item') else item for item in cleaned]
    return cleaned.tolist()

def load_tehran_shapefile(filename):
    shp_path = os.path.join(SRC_DIR, 'polygons', 'tehran_districts', filename)
    tried_encodings = [None, 'cp1256', 'utf-8']
    loaded_gdf = None
    for enc in tried_encodings:
        try:
            gdf_temp = gpd.read_file(shp_path, encoding=enc if enc else None)
            print(f"Loaded {filename} using encoding='{enc or 'default'}'")
            loaded_gdf = gdf_temp
            break
        except Exception as e:
            print(f"  • Trying encoding {enc!r} for {filename}: failed with: {e}")
    if loaded_gdf is None:
        print(f"Could not load {filename}")
        return gpd.GeoDataFrame()
    if loaded_gdf.crs and loaded_gdf.crs.to_string() != "EPSG:4326":
        try:
            loaded_gdf = loaded_gdf.to_crs("EPSG:4326")
            print(f"  • Reprojected {filename} to EPSG:4326")
        except Exception as e:
            print(f"  • Failed to reproject {filename}: {e}")
    print(f"{filename} columns: {list(loaded_gdf.columns)}")
    return loaded_gdf

def get_district_names_from_gdf(gdf, default_prefix="District"):
    if gdf is None or gdf.empty: return []
    name_cols = ['Name', 'name', 'NAME', 'Region', 'REGION_N', 'NAME_MAHAL', 'NAME_1', 'NAME_2', 'district']
    for col in name_cols:
        if col in gdf.columns and gdf[col].dtype == 'object':
            return sorted(safe_tolist(gdf[col].astype(str)))
    for col in gdf.columns: # Fallback
        if col != 'geometry' and gdf[col].dtype == 'object':
            return sorted(safe_tolist(gdf[col].astype(str)))
    return [f"{default_prefix} {i+1}" for i in range(len(gdf))]

def generate_random_points_in_polygon(poly, num_points):
    """Generates a specified number of random points within a given Shapely polygon."""
    points = []
    min_x, min_y, max_x, max_y = poly.bounds
    while len(points) < num_points:
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if random_point.within(poly):
            points.append(random_point)
    return points

def generate_coverage_grid(city_name, grid_size_meters=200):
    """Generate a grid of points for coverage analysis."""
    if city_name not in city_boundaries:
        return []
    
    bounds = city_boundaries[city_name]
    
    # Convert grid size from meters to approximate degrees
    # At these latitudes, 1 degree ≈ 111 km
    grid_size_deg = grid_size_meters / 111000.0
    
    grid_points = []
    lat = bounds["min_lat"]
    while lat <= bounds["max_lat"]:
        lng = bounds["min_lng"]
        while lng <= bounds["max_lng"]:
            grid_points.append({"lat": lat, "lng": lng})
            lng += grid_size_deg
        lat += grid_size_deg
    
    return grid_points

def calculate_coverage_for_grid_vectorized(grid_points, df_vendors_filtered, city_name):
    """
    Calculate vendor coverage for all grid points using vectorized operations.
    Much faster than calculating point by point.
    """
    if df_vendors_filtered.empty or not grid_points:
        return []
    
    # Filter vendors with valid data
    valid_vendors = df_vendors_filtered.dropna(subset=['latitude', 'longitude', 'radius'])
    if valid_vendors.empty:
        return []
    
    # Convert to numpy arrays for faster computation
    grid_lats = np.array([p['lat'] for p in grid_points])
    grid_lngs = np.array([p['lng'] for p in grid_points])
    
    vendor_lats = valid_vendors['latitude'].values
    vendor_lngs = valid_vendors['longitude'].values
    vendor_radii = valid_vendors['radius'].values * 1000  # Convert km to meters
    
    # Pre-extract vendor attributes
    if 'business_line' in valid_vendors and pd.api.types.is_categorical_dtype(valid_vendors['business_line']):
        vendor_business_lines = valid_vendors['business_line'].cat.add_categories(['Unknown']).fillna('Unknown').values
    else:
        vendor_business_lines = valid_vendors['business_line'].fillna('Unknown').values if 'business_line' in valid_vendors else None
    if 'grade' in valid_vendors and pd.api.types.is_categorical_dtype(valid_vendors['grade']):
        vendor_grades = valid_vendors['grade'].cat.add_categories(['Unknown']).fillna('Unknown').values
    else:
        vendor_grades = valid_vendors['grade'].fillna('Unknown').values if 'grade' in valid_vendors else None
    
    coverage_results = []
    
    # Process in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(grid_points), batch_size):
        batch_end = min(i + batch_size, len(grid_points))
        batch_lats = grid_lats[i:batch_end]
        batch_lngs = grid_lngs[i:batch_end]
        
        # Vectorized distance calculation for the batch
        # Using broadcasting to calculate distances from all batch points to all vendors
        lat_diff = batch_lats[:, np.newaxis] - vendor_lats[np.newaxis, :]
        lng_diff = batch_lngs[:, np.newaxis] - vendor_lngs[np.newaxis, :]
        
        # Approximate distance in meters (good enough for our scale)
        distances_meters = np.sqrt((lat_diff * 111000)**2 + (lng_diff * 111000 * np.cos(np.radians(batch_lats[:, np.newaxis])))**2)
        
        # Check which vendors cover each point
        coverage_matrix = distances_meters <= vendor_radii[np.newaxis, :]
        
        # Process results for each point in batch
        for j, point_idx in enumerate(range(i, batch_end)):
            covering_vendors = np.where(coverage_matrix[j])[0]
            
            coverage_data = {
                "lat": grid_points[point_idx]['lat'],
                "lng": grid_points[point_idx]['lng'],
                "total_vendors": len(covering_vendors),
                "by_business_line": {},
                "by_grade": {}
            }
            
            if len(covering_vendors) > 0:
                # Count by business line
                if vendor_business_lines is not None:
                    bl_counts = {}
                    for vendor_idx in covering_vendors:
                        bl = vendor_business_lines[vendor_idx]
                        bl_counts[bl] = bl_counts.get(bl, 0) + 1
                    coverage_data["by_business_line"] = bl_counts
                
                # Count by grade
                if vendor_grades is not None:
                    grade_counts = {}
                    for vendor_idx in covering_vendors:
                        grade = vendor_grades[vendor_idx]
                        grade_counts[grade] = grade_counts.get(grade, 0) + 1
                    coverage_data["by_grade"] = grade_counts
            
            coverage_results.append(coverage_data)
    
    return coverage_results

def find_marketing_area_for_points(points, city_name):
    """
    Find which marketing area each point belongs to.
    --- FIXED: Correctly uses STRtree for efficient point-in-polygon queries. ---
    """
    if city_name not in gdf_marketing_areas or gdf_marketing_areas[city_name].empty:
        return [(None, None)] * len(points)
    gdf_areas = gdf_marketing_areas[city_name]
    results = []
    try:
        from shapely.strtree import STRtree
        area_geoms = gdf_areas.geometry.values
        area_ids = gdf_areas['area_id'].values if 'area_id' in gdf_areas else [None] * len(gdf_areas)
        area_names = gdf_areas['name'].values if 'name' in gdf_areas else [None] * len(gdf_areas)
        # Create the spatial index from the polygon geometries
        tree = STRtree(area_geoms)
        for point in points:
            point_geom = Point(point['lng'], point['lat'])
            # Step 1: Query the tree to find candidate polygons (fast BBOX intersection)
            # This returns indices of geometries whose bounding boxes intersect the point's bounding box.
            candidate_indices = tree.query(point_geom)
            found_area = False
            # Step 2: Iterate through only the candidates and perform the exact 'contains' check
            for idx in candidate_indices:
                candidate_poly = area_geoms[idx]
                if candidate_poly.contains(point_geom):
                    # Found a match!
                    results.append((area_ids[idx], area_names[idx]))
                    found_area = True
                    break  # Stop after finding the first containing polygon
            if not found_area:
                # This point was not in any of the candidate polygons
                results.append((None, None))
    except ImportError:
        # Fallback to non-indexed method (slower, but its logic was correct)
        print("Warning: shapely.strtree not found. Using slower point-in-polygon check.")
        for point in points:
            point_geom = Point(point['lng'], point['lat'])
            found = False
            for idx, area in gdf_areas.iterrows():
                if area.geometry.contains(point_geom):
                    area_id = area.get('area_id')
                    area_name = area.get('name')
                    results.append((area_id, area_name))
                    found = True
                    break
            if not found:
                results.append((None, None))
    return results

# NEW: Improved heatmap functions
def remove_outliers_and_normalize_improved(df, value_column, method='robust'):
    """
    Improved outlier removal and normalization using robust statistical methods.
    """
    if df.empty or value_column not in df.columns:
        return df
    
    df_copy = df.copy()
    df_copy = df_copy[df_copy[value_column].notna() & (df_copy[value_column] > 0)]
    
    if df_copy.empty:
        print(f"No valid {value_column} data after removing nulls/zeros")
        return df_copy
    
    values = df_copy[value_column].values
    
    if method == 'robust':
        # Use IQR method for outlier removal (more stable than percentiles)
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Keep some outliers to maintain data integrity
        lower_bound = max(lower_bound, np.percentile(values, 1))
        upper_bound = min(upper_bound, np.percentile(values, 99))
        
    elif method == 'zscore':
        # Z-score method
        z_scores = np.abs(stats.zscore(values))
        threshold = 3
        mask = z_scores < threshold
        df_copy = df_copy[mask]
        values = df_copy[value_column].values
        lower_bound = values.min()
        upper_bound = values.max()
    
    # Apply bounds
    df_copy = df_copy[(df_copy[value_column] >= lower_bound) & (df_copy[value_column] <= upper_bound)]
    
    if df_copy.empty:
        print(f"No data left after outlier removal for {value_column}")
        return df_copy
    
    # Use log transformation for better distribution
    log_values = np.log1p(df_copy[value_column])
    
    # Robust normalization using percentiles instead of min-max
    p5 = np.percentile(log_values, 5)
    p95 = np.percentile(log_values, 95)
    
    if p95 > p5:
        # Scale to 0-100 range, but cap extreme values
        normalized = ((log_values - p5) / (p95 - p5)) * 100
        normalized = np.clip(normalized, 0, 100)  # Ensure values stay in bounds
    else:
        normalized = np.full(len(df_copy), 50)
    
    df_copy[f'{value_column}_normalized'] = normalized
    
    # Also create a raw normalized version for comparison
    raw_values = df_copy[value_column].values
    raw_p5 = np.percentile(raw_values, 5)
    raw_p95 = np.percentile(raw_values, 95)
    
    if raw_p95 > raw_p5:
        raw_normalized = ((raw_values - raw_p5) / (raw_p95 - raw_p5)) * 100
        raw_normalized = np.clip(raw_normalized, 0, 100)
    else:
        raw_normalized = np.full(len(df_copy), 50)
    
    df_copy[f'{value_column}_raw_normalized'] = raw_normalized
    
    print(f"{value_column} normalization complete: {len(df_copy)} points")
    print(f"Value range: {df_copy[value_column].min():.2f} - {df_copy[value_column].max():.2f}")
    print(f"Normalized range: {normalized.min():.2f} - {normalized.max():.2f}")
    
    return df_copy

def aggregate_heatmap_points_adaptive(df, lat_col, lng_col, value_col, zoom_level=11):
    """
    Adaptive aggregation that adjusts precision based on zoom level and data density.
    """
    if df.empty:
        return df
    
    df_copy = df.copy()
    
    # Adaptive precision based on zoom level
    # Higher zoom = more precision, lower zoom = less precision
    if zoom_level >= 16:
        precision = 5  # Very fine
    elif zoom_level >= 14:
        precision = 4  # Fine
    elif zoom_level >= 12:
        precision = 3  # Medium
    elif zoom_level >= 10:
        precision = 2  # Coarse
    else:
        precision = 1  # Very coarse
    
    # Round coordinates
    df_copy['lat_rounded'] = df_copy[lat_col].round(precision)
    df_copy['lng_rounded'] = df_copy[lng_col].round(precision)
    
    # Aggregate with multiple statistics
    aggregated = df_copy.groupby(['lat_rounded', 'lng_rounded']).agg({
        value_col: ['sum', 'count', 'mean']
    }).reset_index()
    
    # Flatten column names
    aggregated.columns = ['lat', 'lng', 'value_sum', 'value_count', 'value_mean']
    
    # Use sum as primary value, but keep count for density weighting
    aggregated['value'] = aggregated['value_sum']
    aggregated['density_weight'] = np.log1p(aggregated['value_count'])  # Log scale for better distribution
    
    # Apply density weighting to reduce noise in sparse areas
    aggregated['weighted_value'] = aggregated['value'] * (1 + aggregated['density_weight'] * 0.1)
    
    return aggregated[['lat', 'lng', 'weighted_value']].rename(columns={'weighted_value': 'value'})

def aggregate_user_heatmap_points_improved(df, lat_col, lng_col, user_col, zoom_level=11):
    """
    Improved user aggregation with better handling of unique users.
    """
    if df.empty:
        return df
    
    df_copy = df.copy()
    
    # Adaptive precision
    if zoom_level >= 16:
        precision = 5
    elif zoom_level >= 14:
        precision = 4
    elif zoom_level >= 12:
        precision = 3
    else:
        precision = 2
    
    df_copy['lat_rounded'] = df_copy[lat_col].round(precision)
    df_copy['lng_rounded'] = df_copy[lng_col].round(precision)
    
    # Count unique users per location
    aggregated = df_copy.groupby(['lat_rounded', 'lng_rounded'])[user_col].nunique().reset_index()
    aggregated.columns = ['lat', 'lng', 'unique_users']
    
    # Apply log transformation for better distribution
    aggregated['value'] = np.log1p(aggregated['unique_users']) * 10  # Scale up for visibility
    
    return aggregated[['lat', 'lng', 'value']]

def generate_improved_heatmap_data(heatmap_type_req, df_orders_filtered, zoom_level=11):
    """
    Generate heatmap data using improved aggregation and normalization.
    """
    heatmap_data = []
    
    if heatmap_type_req not in ["order_density", "order_density_organic", "order_density_non_organic", "user_density"]:
        return heatmap_data
    
    df_hm_source = df_orders_filtered.dropna(subset=['customer_latitude', 'customer_longitude'])
    if df_hm_source.empty:
        return heatmap_data
    
    if heatmap_type_req == "order_density":
        df_hm_source['order_count'] = 1
        df_aggregated = aggregate_heatmap_points_adaptive(
            df_hm_source, 'customer_latitude', 'customer_longitude', 'order_count', zoom_level
        )
    elif heatmap_type_req == "order_density_organic":
        if 'organic' in df_hm_source.columns:
            df_organic = df_hm_source[df_hm_source['organic'] == 1]
            if not df_organic.empty:
                df_aggregated = aggregate_heatmap_points_adaptive(
                    df_organic.assign(order_count=1), 'customer_latitude', 'customer_longitude', 'order_count', zoom_level
                )
            else:
                df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
        else:
            df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
    elif heatmap_type_req == "order_density_non_organic":
        if 'organic' in df_hm_source.columns:
            df_non_organic = df_hm_source[df_hm_source['organic'] == 0]
            if not df_non_organic.empty:
                df_aggregated = aggregate_heatmap_points_adaptive(
                    df_non_organic.assign(order_count=1), 'customer_latitude', 'customer_longitude', 'order_count', zoom_level
                )
            else:
                df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
        else:
            df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
    elif heatmap_type_req == "user_density":
        if 'user_id' in df_hm_source.columns:
            df_aggregated = aggregate_user_heatmap_points_improved(
                df_hm_source, 'customer_latitude', 'customer_longitude', 'user_id', zoom_level
            )
        else:
            df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
    
    # Apply improved normalization
    if not df_aggregated.empty:
        df_normalized = remove_outliers_and_normalize_improved(df_aggregated, 'value', method='robust')
        if not df_normalized.empty:
            # Use the normalized values
            df_normalized['value'] = df_normalized['value_normalized']
            heatmap_data = df_normalized[['lat', 'lng', 'value']].to_dict(orient='records')
    
    return heatmap_data

def generate_basic_heatmap_fallback(heatmap_type_req, df_orders_filtered):
    """
    Fallback method using the original heatmap generation logic.
    """
    try:
        df_hm_source = df_orders_filtered.dropna(subset=['customer_latitude', 'customer_longitude'])
        if df_hm_source.empty:
            return []
        
        if heatmap_type_req == "order_density":
            df_hm_source['order_count'] = 1
            df_aggregated = aggregate_heatmap_points(df_hm_source, 'customer_latitude', 'customer_longitude', 'order_count', precision=4)
        elif heatmap_type_req == "order_density_organic":
            if 'organic' in df_hm_source.columns:
                df_organic = df_hm_source[df_hm_source['organic'] == 1]
                df_aggregated = aggregate_heatmap_points(df_organic.assign(order_count=1), 'customer_latitude', 'customer_longitude', 'order_count', precision=4) if not df_organic.empty else pd.DataFrame(columns=['lat', 'lng', 'value'])
            else: 
                df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
        elif heatmap_type_req == "order_density_non_organic":
            if 'organic' in df_hm_source.columns:
                df_non_organic = df_hm_source[df_hm_source['organic'] == 0]
                df_aggregated = aggregate_heatmap_points(df_non_organic.assign(order_count=1), 'customer_latitude', 'customer_longitude', 'order_count', precision=4) if not df_non_organic.empty else pd.DataFrame(columns=['lat', 'lng', 'value'])
            else: 
                df_aggregated = pd.DataFrame(columns=['lat', 'lng', 'value'])
        elif heatmap_type_req == "user_density":
            df_aggregated = aggregate_user_heatmap_points(df_hm_source, 'customer_latitude', 'customer_longitude', 'user_id', precision=4) if 'user_id' in df_hm_source.columns else pd.DataFrame(columns=['lat', 'lng', 'value'])
        
        if not df_aggregated.empty and heatmap_type_req != "user_density":
            max_count = df_aggregated['value'].max()
            min_count = df_aggregated['value'].min()
            df_aggregated['value'] = ((df_aggregated['value'] - min_count) / (max_count - min_count)) * 100 if max_count > min_count else 50
        
        return df_aggregated.to_dict(orient='records')
        
    except Exception as e:
        print(f"Error in fallback heatmap generation: {e}")
        return []

# LEGACY: Keep original functions for backward compatibility
def remove_outliers_and_normalize(df, value_column, lower_percentile=5, upper_percentile=95):
    """
    Remove outliers using percentile method and normalize values to 0-100 range.
    Returns a copy of the dataframe with normalized values.
    """
    if df.empty or value_column not in df.columns:
        return df
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Remove rows where the value is null or zero
    df_copy = df_copy[df_copy[value_column].notna() & (df_copy[value_column] > 0)]
    
    if df_copy.empty:
        print(f"No valid {value_column} data after removing nulls/zeros")
        return df_copy
    
    # Calculate percentiles for outlier removal
    lower_bound = df_copy[value_column].quantile(lower_percentile / 100)
    upper_bound = df_copy[value_column].quantile(upper_percentile / 100)
    
    print(f"{value_column} bounds: {lower_bound:,.0f} to {upper_bound:,.0f}")
    
    # Remove outliers
    df_copy = df_copy[(df_copy[value_column] >= lower_bound) & (df_copy[value_column] <= upper_bound)]
    
    if df_copy.empty:
        print(f"No data left after outlier removal for {value_column}")
        return df_copy
    
    # Normalize to 0-100 range
    min_val = df_copy[value_column].min()
    max_val = df_copy[value_column].max()
    
    if max_val > min_val:
        df_copy[f'{value_column}_normalized'] = ((df_copy[value_column] - min_val) / (max_val - min_val)) * 100
    else:
        df_copy[f'{value_column}_normalized'] = 50  # If all values are the same, set to middle value
    
    # Log transformation for better distribution (optional, helps with skewed data)
    # This helps when you have many small values and few large values
    df_copy[f'{value_column}_log_normalized'] = np.log1p(df_copy[value_column])
    log_min = df_copy[f'{value_column}_log_normalized'].min()
    log_max = df_copy[f'{value_column}_log_normalized'].max()
    
    if log_max > log_min:
        df_copy[f'{value_column}_log_normalized'] = ((df_copy[f'{value_column}_log_normalized'] - log_min) / (log_max - log_min)) * 100
    else:
        df_copy[f'{value_column}_log_normalized'] = 50
    
    print(f"{value_column} normalization complete: {len(df_copy)} points")
    return df_copy

def aggregate_heatmap_points(df, lat_col, lng_col, value_col, precision=None):
    """
    Aggregate heatmap points by rounding coordinates to create heat accumulation.
    This ensures areas with multiple orders show more heat.
    """
    if df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Round coordinates to aggregate nearby points
    df_copy['lat_rounded'] = df_copy[lat_col].round(precision)
    df_copy['lng_rounded'] = df_copy[lng_col].round(precision)
    
    # Aggregate values for the same rounded location
    aggregated = df_copy.groupby(['lat_rounded', 'lng_rounded']).agg({
        value_col: 'sum'
    }).reset_index()
    
    # Rename columns to standard names for output
    aggregated['lat'] = aggregated['lat_rounded']
    aggregated['lng'] = aggregated['lng_rounded']
    aggregated['value'] = aggregated[value_col]
    
    return aggregated[['lat', 'lng', 'value']]

def aggregate_user_heatmap_points(df, lat_col, lng_col, user_col, precision=4):
    """
    Aggregate unique users by location for user heatmap.
    """
    if df.empty:
        return df
    
    # Create a copy
    df_copy = df.copy()
    
    # Round coordinates
    df_copy['lat_rounded'] = df_copy[lat_col].round(precision)
    df_copy['lng_rounded'] = df_copy[lng_col].round(precision)
    
    # Count unique users per location
    aggregated = df_copy.groupby(['lat_rounded', 'lng_rounded'])[user_col].nunique().reset_index()
    
    # Rename columns
    aggregated['lat'] = aggregated['lat_rounded']
    aggregated['lng'] = aggregated['lng_rounded']
    aggregated['value'] = aggregated[user_col]
    
    # Normalize values
    if len(aggregated) > 0:
        max_val = aggregated['value'].max()
        min_val = aggregated['value'].min()
        if max_val > min_val:
            aggregated['value'] = ((aggregated['value'] - min_val) / (max_val - min_val)) * 100
        else:
            aggregated['value'] = 50
    
    return aggregated[['lat', 'lng', 'value']]

# REMOVED: Synchronous load_data function - now handled by scheduler
# Data loading is now managed by the DataScheduler in background threads

# --- Helper Functions for Data Loading (kept for compatibility) ---

# --- Serve Static Files (Frontend) ---
@app.route('/')
def serve_index():
    return send_from_directory(config.app.public_dir, 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        data_status = data_scheduler.get_data_status()
        cache_stats = cache_manager.get_stats()
        cache_healthy = cache_manager.health_check()
        
        # Determine overall health
        healthy = (
            data_status['orders']['loaded'] or data_status['vendors']['loaded']
        ) and app_initialized
        
        status_code = 200 if healthy else 503
        
        return jsonify({
            'status': 'healthy' if healthy else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0-production',
            'data_status': data_status,
            'cache_stats': cache_stats,
            'cache_healthy': cache_healthy,
            'scheduler_running': data_scheduler.scheduler is not None and data_scheduler.scheduler.running
        }), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/api/initial-data', methods=['GET'])
@cache_decorator.cache_result(prefix='initial_data', ttl=1800)  # Cache for 30 minutes
def get_initial_data():
    """Get all initial data for the dashboard"""
    try:
        # Get data from scheduler
        df_orders = data_scheduler.get_orders_data()
        df_vendors = data_scheduler.get_vendors_data()
        static_data = data_scheduler.get_static_data()
        
        if df_orders is None and df_vendors is None:
            return jsonify({
                'error': 'Data is still loading. Please wait a moment and try again.',
                'loading': True
            }), 202
        
        gdf_marketing_areas = static_data['marketing_areas']
        gdf_tehran_region = static_data['tehran_region']
        gdf_tehran_main_districts = static_data['tehran_districts']
        df_coverage_targets = static_data['coverage_targets']
        target_lookup_dict = static_data['target_lookup']
        
        cities = [{"id": cid, "name": name} for cid, name in city_id_map.items()]
        business_lines = []
        if df_orders is not None and not df_orders.empty and 'business_line' in df_orders.columns:
            business_lines = safe_tolist(df_orders['business_line'])
        
        marketing_area_names_by_city = {}
        for city_key, gdf in gdf_marketing_areas.items():
            if not gdf.empty and 'name' in gdf.columns:
                marketing_area_names_by_city[city_key] = sorted(safe_tolist(gdf['name'].astype(str)))
            else: 
                marketing_area_names_by_city[city_key] = []
        
        tehran_region_districts = get_district_names_from_gdf(gdf_tehran_region, "Region Tehran")
        tehran_main_districts = get_district_names_from_gdf(gdf_tehran_main_districts, "Main Tehran")
        
        vendor_statuses = []
        if df_vendors is not None and not df_vendors.empty and 'status_id' in df_vendors.columns:
            # Filter out NaN values before converting to int
            status_series = df_vendors['status_id'].dropna()
            if not status_series.empty:
                vendor_statuses = sorted([int(x) for x in status_series.unique()])
        
        vendor_grades = []
        if df_vendors is not None and not df_vendors.empty and 'grade' in df_vendors.columns:
            vendor_grades = sorted(safe_tolist(df_vendors['grade'].astype(str)))
        
        return jsonify({
            "cities": cities,
            "business_lines": business_lines,
            "marketing_areas_by_city": marketing_area_names_by_city,
            "tehran_region_districts": tehran_region_districts,
            "tehran_main_districts": tehran_main_districts,
            "vendor_statuses": vendor_statuses,
            "vendor_grades": vendor_grades
        })
        
    except Exception as e:
        logger.error(f"Error in get_initial_data: {e}")
        return jsonify({
            'error': 'Failed to load initial data',
            'details': str(e)
        }), 500

def enrich_polygons_with_stats(gdf_polygons, name_col, df_v_filtered, df_o_filtered, df_o_all_for_city):
    """
    Enriches a polygon GeoDataFrame with vendor and user statistics.
    Args:
        gdf_polygons (gpd.GeoDataFrame): The polygons to enrich.
        name_col (str): The name of the unique identifier column in the polygon GDF.
        df_v_filtered (pd.DataFrame): Pre-filtered vendor data.
        df_o_filtered (pd.DataFrame): Pre-filtered order data (by date, bl, etc.).
        df_o_all_for_city (pd.DataFrame): Order data filtered only by city (for total user count).
    Returns:
        gpd.GeoDataFrame: The enriched GeoDataFrame.
    """
    if gdf_polygons is None or gdf_polygons.empty:
        return gpd.GeoDataFrame()
    enriched_gdf = gdf_polygons.copy()
    # --- 1. Vendor Enrichment ---
    if not df_v_filtered.empty and not df_v_filtered.dropna(subset=['latitude', 'longitude']).empty:
        gdf_v_filtered_for_enrich = gpd.GeoDataFrame(
            df_v_filtered.dropna(subset=['latitude', 'longitude']),
            geometry=gpd.points_from_xy(df_v_filtered.longitude, df_v_filtered.latitude),
            crs="EPSG:4326"
        )
        joined_vendors = gpd.sjoin(gdf_v_filtered_for_enrich, enriched_gdf, how="inner", predicate="within")
        # Total vendor count
        vendor_counts = joined_vendors.groupby(name_col).size().rename('vendor_count')
        enriched_gdf = enriched_gdf.merge(vendor_counts, how='left', left_on=name_col, right_index=True)
        # Vendor count by grade
        if 'grade' in joined_vendors.columns:
            grade_counts_series = joined_vendors.groupby([name_col, 'grade'], observed=True).size().unstack(fill_value=0)
            grade_counts_dict = grade_counts_series.apply(lambda row: {k: v for k, v in row.items() if v > 0}, axis=1).to_dict()
            enriched_gdf['grade_counts'] = enriched_gdf[name_col].astype(str).map(grade_counts_dict)
        else:
             enriched_gdf['grade_counts'] = None
    else:
        enriched_gdf['vendor_count'] = 0
        enriched_gdf['grade_counts'] = None
    
    enriched_gdf['vendor_count'] = enriched_gdf['vendor_count'].fillna(0).astype(int)
    # --- 2. Unique User Enrichment (Date-Ranged & Total) ---
    has_user_id = 'user_id' in df_o_all_for_city.columns
    if has_user_id:
        # A) Date-Ranged Unique Users
        if not df_o_filtered.empty and not df_o_filtered.dropna(subset=['customer_latitude', 'customer_longitude']).empty:
            gdf_orders_filtered = gpd.GeoDataFrame(
                df_o_filtered.dropna(subset=['customer_latitude', 'customer_longitude']),
                geometry=gpd.points_from_xy(df_o_filtered.customer_longitude, df_o_filtered.customer_latitude),
                crs="EPSG:4326"
            )
            joined_orders_filtered = gpd.sjoin(gdf_orders_filtered, enriched_gdf, how="inner", predicate="within")
            user_counts_filtered = joined_orders_filtered.groupby(name_col, observed=True)['user_id'].nunique().rename('unique_user_count')
            enriched_gdf = enriched_gdf.merge(user_counts_filtered, how='left', left_on=name_col, right_index=True)
        
        # B) Total (All-Time) Unique Users for the city
        if not df_o_all_for_city.empty and not df_o_all_for_city.dropna(subset=['customer_latitude', 'customer_longitude']).empty:
            gdf_orders_all = gpd.GeoDataFrame(
                df_o_all_for_city.dropna(subset=['customer_latitude', 'customer_longitude']),
                geometry=gpd.points_from_xy(df_o_all_for_city.customer_longitude, df_o_all_for_city.customer_latitude),
                crs="EPSG:4326"
            )
            joined_orders_all = gpd.sjoin(gdf_orders_all, enriched_gdf, how="inner", predicate="within")
            user_counts_all = joined_orders_all.groupby(name_col, observed=True)['user_id'].nunique().rename('total_unique_user_count')
            enriched_gdf = enriched_gdf.merge(user_counts_all, how='left', left_on=name_col, right_index=True)
    enriched_gdf['unique_user_count'] = enriched_gdf.get('unique_user_count', pd.Series(0, index=enriched_gdf.index)).fillna(0).astype(int)
    enriched_gdf['total_unique_user_count'] = enriched_gdf.get('total_unique_user_count', pd.Series(0, index=enriched_gdf.index)).fillna(0).astype(int)
    
    # --- 3. Population-based Metrics (if Pop data exists) ---
    if 'Pop' in enriched_gdf.columns:
        enriched_gdf['Pop'] = pd.to_numeric(enriched_gdf['Pop'], errors='coerce').fillna(0)
        enriched_gdf['vendor_per_10k_pop'] = enriched_gdf.apply(
            lambda row: (row['vendor_count'] / row['Pop']) * 10000 if row['Pop'] > 0 else 0, axis=1
        )
    if 'PopDensity' in enriched_gdf.columns:
        enriched_gdf['PopDensity'] = pd.to_numeric(enriched_gdf['PopDensity'], errors='coerce').fillna(0)
        
    return enriched_gdf

@app.route('/api/map-data', methods=['GET'])
@cache_decorator.cache_heatmap(ttl=1800)  # Cache heatmaps for 30 minutes
def get_map_data():
    """Optimized endpoint for map visualization data"""
    try:
        # Get fresh data from scheduler
        df_orders = data_scheduler.get_orders_data()
        df_vendors = data_scheduler.get_vendors_data()
        static_data = data_scheduler.get_static_data()
        
        if df_orders is None or df_vendors is None:
            return jsonify({
                'error': 'Data is still loading. Please wait a moment and try again.',
                'loading': True
            }), 202
        
        gdf_marketing_areas = static_data['marketing_areas']
        
        # Start timing
        request_start = time.time()
        
        # --- Parsing of filters ---
        city_name = request.args.get('city', default="tehran", type=str)
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        start_date = pd.to_datetime(start_date_str) if start_date_str else None
        end_date = pd.to_datetime(end_date_str).replace(hour=23, minute=59, second=59) if end_date_str else None
        selected_business_lines = [bl.strip() for bl in request.args.getlist('business_lines') if bl.strip()]
        vendor_codes_input_str = request.args.get('vendor_codes_filter', default="", type=str)
        selected_vendor_codes_for_vendors_only = [code.strip() for code in re.split(r'[\s,;\n]+', vendor_codes_input_str) if code.strip()]
        selected_vendor_status_ids = [int(s.strip()) for s in request.args.getlist('vendor_status_ids') if s.strip().isdigit()]
        selected_vendor_grades = [g.strip() for g in request.args.getlist('vendor_grades') if g.strip()]
        vendor_visible_str = request.args.get('vendor_visible', default="any", type=str)
        vendor_is_open_str = request.args.get('vendor_is_open', default="any", type=str)
        vendor_area_main_type = request.args.get('vendor_area_main_type', default="all", type=str)
        selected_vendor_area_sub_types = [s.strip() for s in request.args.getlist('vendor_area_sub_type') if s.strip()]
        heatmap_type_req = request.args.get('heatmap_type_request', default="none", type=str)
        area_type_display = request.args.get('area_type_display', default="tapsifood_marketing_areas", type=str)
        selected_polygon_sub_types = [s.strip() for s in request.args.getlist('area_sub_type_filter') if s.strip()]
        
        # NEW: Get zoom level for adaptive heatmap processing
        zoom_level = request.args.get('zoom_level', default=11, type=float)
        
        # Radius modifier parameters
        radius_modifier = request.args.get('radius_modifier', default=1.0, type=float)
        radius_mode = request.args.get('radius_mode', default='percentage', type=str)
        radius_fixed = request.args.get('radius_fixed', default=3.0, type=float)
        
        heatmap_data = []
        vendor_markers = []
        polygons_geojson = {"type": "FeatureCollection", "features": []}
        coverage_grid_data = []
        
        # --- Vendor filtering logic ---
        df_v_filtered = df_vendors.copy()
        
        # Check if required columns exist
        required_vendor_columns = ['latitude', 'longitude', 'vendor_code']
        missing_columns = [col for col in required_vendor_columns if col not in df_v_filtered.columns]
        if missing_columns:
            print(f"Warning: Missing vendor columns: {missing_columns}")
            vendor_markers = []
        else:
            # Apply radius modifier based on mode
            if 'radius' in df_v_filtered.columns and 'original_radius' in df_v_filtered.columns:
                if radius_mode == 'fixed':
                    df_v_filtered['radius'] = radius_fixed
                else:
                    df_v_filtered['radius'] = df_v_filtered['original_radius'] * radius_modifier
            
            # Only filter if columns exist
            df_v_filtered = df_v_filtered.dropna(subset=['latitude', 'longitude', 'vendor_code'])
            
            if city_name != "all" and 'city_name' in df_v_filtered.columns: 
                df_v_filtered = df_v_filtered[df_v_filtered['city_name'] == city_name]
            
            if selected_vendor_codes_for_vendors_only and 'vendor_code' in df_v_filtered.columns: 
                df_v_filtered = df_v_filtered[df_v_filtered['vendor_code'].astype(str).isin(selected_vendor_codes_for_vendors_only)]
            
            if vendor_area_main_type != 'all' and selected_vendor_area_sub_types and not df_v_filtered.empty:
                if vendor_area_main_type == 'tapsifood_marketing_areas' and not df_orders.empty and 'marketing_area' in df_orders.columns:
                    temp_orders_ma = df_orders[df_orders['marketing_area'].isin(selected_vendor_area_sub_types)]
                    relevant_vendor_codes_ma = temp_orders_ma['vendor_code'].astype(str).dropna().unique()
                    df_v_filtered = df_v_filtered[df_v_filtered['vendor_code'].isin(relevant_vendor_codes_ma)]
                elif city_name == 'tehran':
                    target_gdf = None
                    if vendor_area_main_type == 'tehran_region_districts': target_gdf = gdf_tehran_region.copy() if gdf_tehran_region is not None else None
                    elif vendor_area_main_type == 'tehran_main_districts': target_gdf = gdf_tehran_main_districts.copy() if gdf_tehran_main_districts is not None else None
                    if target_gdf is not None and not target_gdf.empty:
                        name_col = next((col for col in ['Name', 'NAME_MAHAL'] if col in target_gdf.columns), None)
                        if name_col: target_gdf = target_gdf[target_gdf[name_col].isin(selected_vendor_area_sub_types)]
                        if not target_gdf.empty:
                            gdf_vendors_to_filter = gpd.GeoDataFrame(df_v_filtered, geometry=gpd.points_from_xy(df_v_filtered.longitude, df_v_filtered.latitude), crs="EPSG:4326")
                            vendors_in_area = gpd.sjoin(gdf_vendors_to_filter, target_gdf, how="inner", predicate="within")
                            codes_in_area = vendors_in_area['vendor_code'].unique() if not vendors_in_area.empty else []
                            df_v_filtered = df_v_filtered[df_v_filtered['vendor_code'].isin(codes_in_area)]
            
            if not df_v_filtered.empty and selected_business_lines and 'business_line' in df_v_filtered.columns:
                df_v_filtered = df_v_filtered[df_v_filtered['business_line'].isin(selected_business_lines)]
            
            if not df_v_filtered.empty:
                if selected_vendor_status_ids and 'status_id' in df_v_filtered.columns: 
                    df_v_filtered = df_v_filtered[df_v_filtered['status_id'].isin(selected_vendor_status_ids)]
                if selected_vendor_grades and 'grade' in df_v_filtered.columns: 
                    df_v_filtered = df_v_filtered[df_v_filtered['grade'].isin(selected_vendor_grades)]
                if vendor_visible_str != "any" and 'visible' in df_v_filtered.columns: 
                    df_v_filtered = df_v_filtered[df_v_filtered['visible'] == int(vendor_visible_str)]
                if vendor_is_open_str != "any" and 'open' in df_v_filtered.columns: 
                    df_v_filtered = df_v_filtered[df_v_filtered['open'] == int(vendor_is_open_str)]
            
            if not df_v_filtered.empty:
                vendor_markers = df_v_filtered.replace({np.nan: None}).to_dict(orient='records')
            else:
                vendor_markers = []
        
        # --- Prepare filtered and total order dataframes for enrichment ---
        df_orders_filtered = df_orders.copy()
        if city_name != "all": df_orders_filtered = df_orders_filtered[df_orders_filtered['city_name'] == city_name]
        df_orders_all_for_city = df_orders_filtered.copy()
        if start_date: df_orders_filtered = df_orders_filtered[df_orders_filtered['created_at'] >= start_date]
        if end_date: df_orders_filtered = df_orders_filtered[df_orders_filtered['created_at'] <= end_date]
        if selected_business_lines: df_orders_filtered = df_orders_filtered[df_orders_filtered['business_line'].isin(selected_business_lines)]
        
        # --- IMPROVED Heatmap generation using new functions ---
        if heatmap_type_req in ["order_density", "order_density_organic", "order_density_non_organic", "user_density"]:
            print(f"Generating improved heatmap for type: {heatmap_type_req} at zoom level: {zoom_level}")
            
            try:
                heatmap_data_result = generate_improved_heatmap_data(heatmap_type_req, df_orders_filtered, zoom_level)
                
                if heatmap_data_result:
                    # Add metadata for frontend optimization
                    print(f"Generated {len(heatmap_data_result)} heatmap points")
                    
                    # Calculate some statistics for the frontend
                    values = [point['value'] for point in heatmap_data_result if 'value' in point and point['value'] is not None]
                    if values:
                        heatmap_stats = {
                            'count': len(values),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'mean': float(np.mean(values)),
                            'p75': float(np.percentile(values, 75)),
                            'p90': float(np.percentile(values, 90)),
                            'p95': float(np.percentile(values, 95))
                        }
                        print(f"Heatmap stats: {heatmap_stats}")
                    
                    heatmap_data = heatmap_data_result
                else:
                    print("No heatmap data generated")
                    heatmap_data = []
                    
            except Exception as e:
                print(f"Error generating improved heatmap: {e}")
                # Fallback to basic method if improved method fails
                heatmap_data = generate_basic_heatmap_fallback(heatmap_type_req, df_orders_filtered)
        
        elif heatmap_type_req == "population" and city_name == "tehran":
            print("Generating population heatmap...")
            gdf_pop_source = None
            if area_type_display == "tehran_main_districts" and gdf_tehran_main_districts is not None: 
                gdf_pop_source = gdf_tehran_main_districts
            elif area_type_display == "tehran_region_districts" and gdf_tehran_region is not None: 
                gdf_pop_source = gdf_tehran_region
            elif area_type_display == "all_tehran_districts" and gdf_tehran_main_districts is not None: 
                gdf_pop_source = gdf_tehran_main_districts
                
            if gdf_pop_source is not None and 'Pop' in gdf_pop_source.columns:
                if selected_polygon_sub_types:
                    name_cols_poly = ['Name', 'NAME_MAHAL']
                    actual_name_col = next((col for col in name_cols_poly if col in gdf_pop_source.columns), None)
                    if actual_name_col: 
                        gdf_pop_source = gdf_pop_source[gdf_pop_source[actual_name_col].isin(selected_polygon_sub_types)]
                
                # Adjust point density based on zoom level for population heatmap
                base_divisor = 1000
                zoom_multiplier = max(0.1, min(2.0, (zoom_level / 11.0)))  # Scale with zoom
                point_density_divisor = base_divisor / zoom_multiplier
                
                temp_points = []
                for _, row in gdf_pop_source.iterrows():
                    population = pd.to_numeric(row['Pop'], errors='coerce')
                    if pd.notna(population) and population > 0:
                        num_points = int(population / point_density_divisor)
                        if num_points > 0:
                            generated_points = generate_random_points_in_polygon(row['geometry'], num_points)
                            for point in generated_points: 
                                temp_points.append({'lat': point.y, 'lng': point.x, 'value': 1})
                
                heatmap_data = temp_points
                print(f"Generated {len(heatmap_data)} points for population heatmap at zoom {zoom_level}")
        
        # --- Coverage Grid Generation ---
        if area_type_display == "coverage_grid":
            print(f"DEBUG: Generating coverage grid for {city_name}")
            
            vendor_codes_for_cache = sorted(df_v_filtered['vendor_code'].tolist()) if not df_v_filtered.empty and 'vendor_code' in df_v_filtered.columns else []
            cache_key = hashlib.md5(json.dumps({
                'city': city_name,
                'vendor_codes': vendor_codes_for_cache,
                'radius_modifier': radius_modifier, 'radius_mode': radius_mode, 'radius_fixed': radius_fixed,
                'business_lines': sorted(selected_business_lines)
            }, sort_keys=True).encode()).hexdigest()
            
            if cache_key in coverage_cache:
                print(f"DEBUG: Using cached coverage grid")
                coverage_grid_data = coverage_cache[cache_key]
            else:
                grid_points = generate_coverage_grid(city_name)
                point_area_info = find_marketing_area_for_points(grid_points, city_name)
                coverage_results = calculate_coverage_for_grid_vectorized(grid_points, df_v_filtered, city_name)
                
                use_target_based_logic = bool(
                    city_name == "tehran" and
                    len(selected_business_lines) == 1 and
                    target_lookup_dict
                )
                selected_bl_for_target = selected_business_lines[0] if use_target_based_logic else None
                
                print(f"DEBUG: Coverage logic check: Target-based = {use_target_based_logic}")
                if use_target_based_logic:
                    print(f"--- Analyzing for Business Line: '{selected_bl_for_target}'")
                
                temp_coverage_grid_data = []
                for i, coverage in enumerate(coverage_results):
                    if coverage['total_vendors'] > 0:
                        area_id, area_name = point_area_info[i]
                        point_data = {
                            'lat': coverage['lat'], 'lng': coverage['lng'],
                            'coverage': coverage, 'marketing_area': area_name
                        }
                        
                        if use_target_based_logic and area_id is not None:
                            target_key = (area_id, selected_bl_for_target)
                            target_value = target_lookup_dict.get(target_key)
                            
                            if i < 5:
                               print(f"--- Point {i}: area_id='{area_id}', bl='{selected_bl_for_target}'. Target lookup result: {target_value}")
                            if target_value is not None:
                                actual_value = coverage['by_business_line'].get(selected_bl_for_target, 0)
                                point_data['target_business_line'] = selected_bl_for_target
                                point_data['target_value'] = target_value
                                point_data['actual_value'] = actual_value
                                point_data['performance_ratio'] = (actual_value / target_value) if target_value > 0 else 2.0
                        temp_coverage_grid_data.append(point_data)
                
                coverage_grid_data = temp_coverage_grid_data
                
                if len(coverage_cache) > CACHE_SIZE: coverage_cache.clear()
                coverage_cache[cache_key] = coverage_grid_data
                print(f"Filtered to {len(coverage_grid_data)} coverage points with vendors")
        
        # --- Centralized Polygon Display & Enrichment Logic ---
        final_polygons_gdf = None
        if area_type_display != "none" and area_type_display != "coverage_grid":
            gdf_to_enrich, name_col_to_use = None, None
            if area_type_display == "tapsifood_marketing_areas" and city_name in gdf_marketing_areas:
                gdf_to_enrich, name_col_to_use = gdf_marketing_areas[city_name], 'name'
            elif city_name == "tehran":
                if area_type_display == "tehran_region_districts" and gdf_tehran_region is not None:
                    gdf_to_enrich, name_col_to_use = gdf_tehran_region, 'Name'
                elif area_type_display == "tehran_main_districts" and gdf_tehran_main_districts is not None:
                    gdf_to_enrich, name_col_to_use = gdf_tehran_main_districts, 'NAME_MAHAL'
            
            if gdf_to_enrich is not None and not gdf_to_enrich.empty and name_col_to_use is not None:
                final_polygons_gdf = enrich_polygons_with_stats(gdf_to_enrich, name_col_to_use, df_v_filtered, df_orders_filtered, df_orders_all_for_city)
            elif area_type_display == "all_tehran_districts" and city_name == 'tehran':
                enriched_list = []
                if gdf_tehran_region is not None: enriched_list.append(enrich_polygons_with_stats(gdf_tehran_region, 'Name', df_v_filtered, df_orders_filtered, df_orders_all_for_city))
                if gdf_tehran_main_districts is not None: enriched_list.append(enrich_polygons_with_stats(gdf_tehran_main_districts, 'NAME_MAHAL', df_v_filtered, df_orders_filtered, df_orders_all_for_city))
                if enriched_list: final_polygons_gdf = pd.concat(enriched_list, ignore_index=True)
            
            if final_polygons_gdf is not None and not final_polygons_gdf.empty:
                if selected_polygon_sub_types:
                    name_cols_poly = ['name', 'Name', 'NAME_MAHAL']
                    actual_name_col = next((col for col in name_cols_poly if col in final_polygons_gdf.columns), None)
                    if actual_name_col: final_polygons_gdf = final_polygons_gdf[final_polygons_gdf[actual_name_col].astype(str).isin(selected_polygon_sub_types)]
                if not final_polygons_gdf.empty:
                    clean_gdf = final_polygons_gdf.copy()
                    for col in clean_gdf.columns:
                        if col == 'geometry': continue
                        dtype = clean_gdf[col].dtype
                        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
                            clean_gdf[col] = clean_gdf[col].astype(object).where(pd.notna(clean_gdf[col]), None)
                        else:
                            clean_gdf[col] = clean_gdf[col].astype(object).where(pd.notna(clean_gdf[col]), None)
                    polygons_geojson = clean_gdf.__geo_interface__
        
        # Marketing areas overlay for coverage grid
        if area_type_display == "coverage_grid" and city_name in gdf_marketing_areas:
            gdf_to_send = gdf_marketing_areas[city_name].copy()
            if selected_polygon_sub_types and 'name' in gdf_to_send.columns:
                gdf_to_send = gdf_to_send[gdf_to_send['name'].astype(str).isin(selected_polygon_sub_types)]
            if not gdf_to_send.empty:
                clean_gdf = gdf_to_send.copy()
                for col in clean_gdf.columns:
                    if col == 'geometry': continue
                    clean_gdf[col] = clean_gdf[col].astype(object).where(pd.notna(clean_gdf[col]), None)
                polygons_geojson = clean_gdf.__geo_interface__
        
        request_time = time.time() - request_start
        print(f"Request processed in {request_time:.2f}s")
        
        response_data = {
            "vendors": vendor_markers, 
            "heatmap_data": heatmap_data, 
            "polygons": polygons_geojson, 
            "coverage_grid": coverage_grid_data,
            "processing_time": request_time,
            # NEW: Add metadata for frontend optimization
            "zoom_level": zoom_level,
            "heatmap_type": heatmap_type_req
        }
        
        try:
            import ujson
            return app.response_class(ujson.dumps(response_data), mimetype='application/json')
        except ImportError:
            return jsonify(response_data)
            
    except Exception as e:
        import traceback
        print(f"Error in /api/map-data: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# --- Function to Open Browser ---
def open_browser():
    """Opens the web browser to the app's URL after a short delay."""
    time.sleep(1)
    url = f"http://127.0.0.1:{config.app.port}/"
    webbrowser.open_new(url)

# --- Main Execution ---
# --- Application Lifecycle Management ---
def initialize_app():
    """Initialize application components"""
    global app_initialized
    
    if app_initialized:
        return
    
    logger.info("🚀 Initializing Tapsi Food Map Dashboard...")
    
    try:
        # Validate configuration
        if not config.validate():
            logger.error("❌ Configuration validation failed")
            return False
        
        # Start background scheduler for data fetching
        data_scheduler.start()
        logger.info("✅ Background scheduler started")
        
        # Warm cache with initial data (will happen after first data load)
        def warm_cache_after_init():
            time.sleep(10)  # Wait for initial data load
            try:
                warm_cache(data_scheduler)
            except Exception as e:
                logger.warning(f"Cache warming failed: {e}")
        
        threading.Thread(target=warm_cache_after_init, daemon=True).start()
        
        app_initialized = True
        logger.info("✅ Application initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Application initialization failed: {e}")
        return False

def shutdown_app():
    """Cleanup on application shutdown"""
    logger.info("🔄 Shutting down application...")
    try:
        data_scheduler.stop()
        logger.info("✅ Background scheduler stopped")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Register cleanup handlers
atexit.register(shutdown_app)

if __name__ == '__main__':
    # Initialize app components
    if not initialize_app():
        logger.error("Failed to initialize application")
        exit(1)
    
    # Start browser in development
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Flask app
    app.run(
        host=config.app.host,
        port=config.app.port,
        debug=config.app.debug,
        use_reloader=config.app.debug
    )