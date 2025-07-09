import streamlit as st
import ollama
import geopandas as gpd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import leafmap.foliumap as leafmap
import requests
import json
from shapely.geometry import Point, LineString, Polygon
import tempfile
import os
import pandas as pd
import time
import psutil
from datetime import datetime
import re
from matplotlib.colors import LinearSegmentedColormap
import chromadb
from chromadb.config import Settings
from langchain.embeddings import OllamaEmbeddings
from langchain.schema import Document
from rasterio.crs import CRS
from shapely.validation import make_valid
import osmnx as ox
import whitebox
from rasterio.io import MemoryFile
import contextlib
import sys
import logging
from rasterio.windows import Window
from rasterio.enums import Resampling
import numpy as np
from PIL import Image
from rasterio import features
from shapely.geometry import shape
import networkx as nx

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'workflow' not in st.session_state:
    st.session_state.workflow = []
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'error_context' not in st.session_state:
    st.session_state.error_context = {}
if 'user_clarification' not in st.session_state:
    st.session_state.user_clarification = None
if 'retry_count' not in st.session_state:
    st.session_state.retry_count = 0
if 'reasoning_log' not in st.session_state:
    st.session_state.reasoning_log = []
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {}
if 'rag_index' not in st.session_state:
    st.session_state.rag_index = None
if 'alternative_locations' not in st.session_state:
    st.session_state.alternative_locations = []
if 'clarification_visible' not in st.session_state:
    st.session_state.clarification_visible = False
if 'structured_query' not in st.session_state:
    st.session_state.structured_query = None
if 'query_clarification' not in st.session_state:
    st.session_state.query_clarification = None
if 'query_attempts' not in st.session_state:
    st.session_state.query_attempts = 0
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'execution_history' not in st.session_state:
    st.session_state.execution_history = []
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = {}
if 'task_type' not in st.session_state:
    st.session_state.task_type = None
if 'flood_zones' not in st.session_state:
    st.session_state.flood_zones = None
if 'danger_zone_gdf' not in st.session_state:
    st.session_state.danger_zone_gdf = None
if 'safe_zone_gdf' not in st.session_state:
    st.session_state.safe_zone_gdf = None
if 'original_query' not in st.session_state:
    st.session_state.original_query = ""
if 'solar_suitable_areas' not in st.session_state:
    st.session_state.solar_suitable_areas = None
if 'hospital_suitable_areas' not in st.session_state:
    st.session_state.hospital_suitable_areas = None

# Constants
MAX_RETRIES = 3
MAX_CLARIFICATION_ATTEMPTS = 2
MAX_QUERY_ATTEMPTS = 2
MAX_PROCESSING_POINTS = 10000

# Predefined locations for major Indian cities
PREDEFINED_LOCATIONS = {
    "Mumbai": {"name": "Mumbai, India", "latitude": 19.0760, "longitude": 72.8777, "bbox": [72.775, 18.89, 72.975, 19.25]},
    "Chennai": {"name": "Chennai, India", "latitude": 13.0827, "longitude": 80.2707, "bbox": [80.17, 13.00, 80.37, 13.17]},
    "Delhi": {"name": "Delhi, India", "latitude": 28.7041, "longitude": 77.1025, "bbox": [76.84, 28.40, 77.36, 28.90]},
    "Bangalore": {"name": "Bangalore, India", "latitude": 12.9716, "longitude": 77.5946, "bbox": [77.36, 12.80, 77.78, 13.14]},
    "Kolkata": {"name": "Kolkata, India", "latitude": 22.5726, "longitude": 88.3639, "bbox": [88.20, 22.45, 88.50, 22.65]}
}

# Initialize WhiteboxTools
wbt = whitebox.WhiteboxTools()
wbt.set_verbose_mode(False)

# Tool definitions
TOOLS = [
    {
        "name": "geocode",
        "description": "Get coordinates for a place name",
        "parameters": {
            "location": {
                "type": "string",
                "description": "Name of the location to geocode",
                "required": True
            }
        },
        "outputs": {
            "coordinates": {
                "type": "object",
                "description": "Geocoded location with name, coordinates and bbox"
            }
        }
    },
    {
        "name": "visualize_location",
        "description": "Show location on interactive map",
        "parameters": {
            "coordinates": {
                "type": "object",
                "description": "Coordinates from geocode step",
                "required": True
            }
        },
        "outputs": {
            "map": {
                "type": "object",
                "description": "Interactive map visualization"
            }
        }
    },
    {
        "name": "get_dem",
        "description": "Fetch Digital Elevation Model data",
        "parameters": {
            "bbox": {
                "type": "array",
                "description": "Bounding box coordinates [min_lat, max_lat, min_lon, max_lon]",
                "required": True
            },
            "source": {
                "type": "string",
                "description": "Data source (bhuvan, osm, or synthetic)",
                "default": "synthetic"
            }
        },
        "outputs": {
            "dem_path": {
                "type": "string",
                "description": "Path to DEM file"
            }
        }
    },
    {
        "name": "calculate_slope",
        "description": "Compute terrain slope from DEM",
        "parameters": {
            "dem_path": {
                "type": "string",
                "description": "Path to DEM file",
                "required": True
            }
        },
        "outputs": {
            "slope": {
                "type": "array",
                "description": "Slope raster data"
            }
        }
    },
    {
        "name": "flood_risk_model",
        "description": "Generate flood risk model",
        "parameters": {
            "dem_path": {
                "type": "string",
                "description": "Path to DEM file",
                "required": True
            },
            "rainfall_mm": {
                "type": "number",
                "description": "Rainfall amount in mm",
                "default": 150
            }
        },
        "outputs": {
            "flood_mask": {
                "type": "array",
                "description": "Flood risk raster"
            },
            "stats": {
                "type": "object",
                "description": "Flood risk statistics"
            }
        }
    },
    {
        "name": "get_osm_data",
        "description": "Fetch OpenStreetMap data for a location",
        "parameters": {
            "location": {
                "type": "string",
                "description": "Location name",
                "required": True
            },
            "feature_type": {
                "type": "string",
                "description": "Type of feature (substation, hospital, road)",
                "default": "substation"
            }
        },
        "outputs": {
            "geodataframe": {
                "type": "geodataframe",
                "description": "GeoDataFrame with features"
            }
        }
    },
    {
        "name": "buffer_analysis",
        "description": "Create buffer zones around features",
        "parameters": {
            "input_layer": {
                "type": "geodataframe",
                "description": "Input vector layer",
                "required": True
            },
            "distance": {
                "type": "number",
                "description": "Buffer distance in meters",
                "required": True
            }
        },
        "outputs": {
            "buffer_layer": {
                "type": "geodataframe",
                "description": "Buffered features"
            }
        }
    },
    {
        "name": "site_suitability_analysis",
        "description": "Perform multi-criteria site suitability analysis",
        "parameters": {
            "criteria_layers": {
                "type": "array",
                "description": "List of criteria layers with weights",
                "required": True
            },
            "weights": {
                "type": "array",
                "description": "Weights for each criteria layer",
                "required": True
            }
        },
        "outputs": {
            "suitability_map": {
                "type": "geodataframe",
                "description": "Suitability scores"
            }
        }
    },
    {
        "name": "get_coastline_data",
        "description": "Fetch coastline data for a region",
        "parameters": {
            "region": {
                "type": "string",
                "description": "Name of the coastal region",
                "required": True
            }
        },
        "outputs": {
            "coastline_layer": {
                "type": "geodataframe",
                "description": "Coastline vector data"
            }
        }
    },
    {
        "name": "solar_site_selection",
        "description": "Identify suitable solar farm locations",
        "parameters": {
            "slope_layer": {
                "type": "array",
                "description": "Slope data from DEM",
                "required": True
            },
            "substations": {
                "type": "geodataframe",
                "description": "Substation locations",
                "required": True
            },
            "max_slope": {
                "type": "number",
                "description": "Maximum allowed slope in degrees",
                "default": 5
            },
            "buffer_distance": {
                "type": "number",
                "description": "Proximity to substations in km",
                "default": 10
            }
        },
        "outputs": {
            "suitable_areas": {
                "type": "geodataframe",
                "description": "Suitable locations for solar farms"
            }
        }
    },
    {
        "name": "hospital_site_selection",
        "description": "Identify suitable hospital locations",
        "parameters": {
            "existing_hospitals": {
                "type": "geodataframe",
                "description": "Existing hospital locations",
                "required": True
            },
            "roads": {
                "type": "geodataframe",
                "description": "Road network",
                "required": True
            },
            "min_distance": {
                "type": "number",
                "description": "Minimum distance from existing hospitals (km)",
                "default": 5
            }
        },
        "outputs": {
            "suitable_areas": {
                "type": "geodataframe",
                "description": "Suitable locations for new hospitals"
            }
        }
    }
]

def validate_query(structured_query):
    """Validate structured query has required parameters"""
    task = structured_query.get('task', '')
    location = structured_query.get('location', '').strip()
    params = structured_query.get('parameters', {})
    
    # Always require location
    if not location:
        return False, "Location is required"
    
    # Task-specific requirements
    if task == "flood":
        if 'rainfall_mm' not in params:
            return False, "Rainfall amount (mm) is required for flood risk analysis"
    elif task == "solar":
        if 'max_slope' not in params:
            params['max_slope'] = 5
        if 'buffer_distance' not in params:
            params['buffer_distance'] = 10
    elif task == "hospital":
        if 'min_distance' not in params:
            params['min_distance'] = 5
            
    return True, "All required parameters present"

def log_reasoning(step, message):
    """Log Chain-of-Thought reasoning"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "step": step,
        "message": message
    }
    st.session_state.reasoning_log.append(entry)
    logger.info(f"{timestamp} - {step}: {message}")
    return entry

def start_performance_monitor(step_name):
    """Start monitoring performance for a step"""
    return {
        "step": step_name,
        "start_time": time.time(),
        "start_memory": psutil.virtual_memory().used
    }

def end_performance_monitor(monitor):
    """End performance monitoring and record metrics"""
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    
    metrics = {
        "runtime": end_time - monitor["start_time"],
        "memory_used": end_memory - monitor["start_memory"]
    }
    
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {}
    
    st.session_state.performance_metrics[monitor["step"]] = metrics
    return metrics

def auto_crs_conversion(gdf, target_crs="EPSG:4326"):
    """Automatically convert CRS if needed"""
    if gdf.crs is None:
        log_reasoning("CRS Handling", "No CRS found. Assigning default EPSG:4326.")
        gdf.crs = target_crs
        return gdf
    
    if gdf.crs != target_crs:
        log_reasoning("CRS Handling", f"Converting CRS from {gdf.crs} to {target_crs}")
        return gdf.to_crs(target_crs)
    
    return gdf

def validate_geometry(gdf):
    """Validate and repair geometries"""
    invalid_count = 0
    for idx, geom in enumerate(gdf.geometry):
        if not geom.is_valid:
            invalid_count += 1
            gdf.geometry[idx] = make_valid(geom)
    
    if invalid_count > 0:
        log_reasoning("Geometry Validation", f"Repaired {invalid_count} invalid geometries")
    
    return gdf

def geocode(location, region="IN", attempts=3):
    if not location:
        return None
    # rest code ...
    
    predefined = {k.lower(): v for k, v in PREDEFINED_LOCATIONS.items()}
    if location.lower() in predefined:
        return predefined[location.lower()]
        
    ambiguous_terms = ["myarea", "my area", "current location", "user's location", "near me", "user's area"]
    if location.lower() in ambiguous_terms:
        raise ValueError("Ambiguous location specified - please clarify exact location")
        
    params = {
        "q": location,
        "format": "json",
        "countrycodes": "in",
        "addressdetails": 1
    }
    
    for attempt in range(attempts):
        try:
            response = requests.get("https://nominatim.openstreetmap.org/search", 
                                  params=params, 
                                  headers={"User-Agent": "GeospatialAI"}, 
                                  timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    for result in data:
                        if 'display_name' in result and location.lower() in result['display_name'].lower():
                            return format_geocode_result(result, location)
                    
                    return format_geocode_result(data[0], location)
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 1))
                log_reasoning("Geocoding", f"Rate limited. Retrying after {retry_after} seconds")
                time.sleep(retry_after)
                continue
        except Exception as e:
            log_reasoning("Geocoding", f"Error geocoding {location} (attempt {attempt+1}): {str(e)}")
            time.sleep(1)
    
    log_reasoning("Geocoding", f"All attempts failed for location: {location}")
    return None

def format_geocode_result(result, original_query):
    """Format geocoding result consistently"""
    bbox = list(map(float, result['boundingbox'])) if 'boundingbox' in result else [
        float(result['lat']) - 0.1, 
        float(result['lat']) + 0.1, 
        float(result['lon']) - 0.1, 
        float(result['lon']) + 0.1
    ]
    return {
        "name": result.get('display_name', original_query),
        "latitude": float(result['lat']),
        "longitude": float(result['lon']),
        "bbox": bbox,
        "type": result.get('type', ''),
        "class": result.get('class', ''),
        "importance": result.get('importance', 0)
    }

def get_osm_data(location, feature_type):
    """Get OSM data using osmnx with error handling"""
    try:
        if feature_type == "substation":
            tags = {"power": "substation"}
        elif feature_type == "hospital":
            tags = {"amenity": "hospital"}
        elif feature_type == "road":
            return ox.graph_from_place(location, network_type='drive')
        else:
            tags = {"building": True}
            
        gdf = ox.features_from_place(location, tags=tags)
        return gdf
    except Exception as e:
        log_reasoning("OSM Data", f"Error fetching {feature_type} data: {str(e)}")
        if feature_type == "substation":
            points = [Point(73.5, 26.5), Point(73.6, 26.6), Point(73.4, 26.4)]
        elif feature_type == "hospital":
            points = [Point(77.0, 28.0), Point(77.1, 28.1)]
        else:
            points = [Point(72.5, 23.5), Point(72.6, 23.6)]
            
        gdf = gpd.GeoDataFrame(
            geometry=points,
            crs="EPSG:4326"
        )
        return gdf

def get_coastline_data(region):
    """Get coastline data for a region (simulated for Chennai)"""
    log_reasoning("Data Acquisition", f"Fetching coastline data for {region}")
    
    if "Chennai" in region:
        coastline = LineString([
            (80.20, 13.15), (80.22, 13.10), (80.25, 13.05), 
            (80.28, 13.00), (80.30, 12.95), (80.32, 12.90)
        ])
    else:
        coastline = LineString([
            (80.0, 13.0), (80.1, 12.9), (80.2, 12.8), 
            (80.3, 12.7), (80.4, 12.6), (80.5, 12.5)
        ])
    
    coastline_buffer = coastline.buffer(0.02)
    
    gdf = gpd.GeoDataFrame(
        [{"name": f"{region} Coastline"}],
        geometry=[coastline_buffer],
        crs="EPSG:4326"
    )
    return gdf

def safe_raster_processing(func):
    """Decorator for safe raster processing with memory management"""
    def wrapper(*args, **kwargs):
        try:
            if 'bounds' in kwargs:
                minx, miny, maxx, maxy = kwargs['bounds']
                area = (maxx - minx) * (maxy - miny)
                if area > 1:
                    kwargs['resolution_factor'] = 0.5
            
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Raster processing failed: {str(e)}")
            if 'resolution_factor' in kwargs:
                kwargs['resolution_factor'] *= 0.7
                if kwargs['resolution_factor'] < 0.1:
                    raise ValueError("Area too large for processing. Please select a smaller region.")
                return wrapper(*args, **kwargs)
            else:
                kwargs['resolution_factor'] = 0.7
                return wrapper(*args, **kwargs)
    return wrapper

@safe_raster_processing
def get_dem(location, resolution_factor=1.0):
    """Get DEM data with fallback to synthetic"""
    try:
        location_info = geocode(location)
        if not location_info:
            raise ValueError("Location not found")
            
        min_lat, max_lat, min_lon, max_lon = location_info['bbox']
        bounds = (min_lon, min_lat, max_lon, max_lat)
        
        cols = int(100 * resolution_factor)
        rows = int(100 * resolution_factor)
        
        x = np.linspace(min_lon, max_lon, cols)
        y = np.linspace(min_lat, max_lat, rows)
        xx, yy = np.meshgrid(x, y)
        elevation = (
            50 * np.sin(0.1 * xx) * np.cos(0.1 * yy) +
            30 * np.sin(0.05 * xx) * np.cos(0.05 * yy) +
            20 * np.random.rand(rows, cols)
        )
        
        return elevation, bounds
    except Exception as e:
        logger.error(f"DEM generation failed: {str(e)}")
        return np.random.rand(100, 100), (72, 18, 74, 20)

@safe_raster_processing
def calculate_slope(dem_array, bounds, resolution_factor=1.0):
    """Calculate slope using in-memory files"""
    try:
        with MemoryFile() as memfile:
            if resolution_factor < 1.0:
                new_shape = (
                    int(dem_array.shape[0] * resolution_factor),
                    int(dem_array.shape[1] * resolution_factor)
                )
                dem_array = np.array(Image.fromarray(dem_array).resize(new_shape))
            
            with memfile.open(
                driver='GTiff',
                height=dem_array.shape[0],
                width=dem_array.shape[1],
                count=1,
                dtype=dem_array.dtype,
                crs='EPSG:4326',
                transform=rasterio.transform.from_bounds(*bounds, 
                                                        dem_array.shape[1], 
                                                        dem_array.shape[0])
            ) as dataset:
                dataset.write(dem_array, 1)
            
            with MemoryFile() as slope_mem:
                wbt.slope(memfile.name, slope_mem.name, units='degrees')
                with slope_mem.open() as src:
                    return src.read(1), src.profile
    except Exception as e:
        logger.error(f"Slope calculation failed: {str(e)}")
        return np.zeros_like(dem_array), None

@safe_raster_processing
def flood_risk_model(dem_array, rainfall, bounds, resolution_factor=1.0):
    """Flood risk model with enhanced type handling and danger/safe zones"""
    monitor = start_performance_monitor("flood_risk_model")
    
    try:
        rainfall = float(rainfall)
    except (TypeError, ValueError):
        log_reasoning("Parameter Handling", 
                     f"Could not convert rainfall value '{rainfall}' to number. Using default 150mm")
        rainfall = 150.0
    
    if resolution_factor < 1.0:
        new_shape = (
            int(dem_array.shape[0] * resolution_factor),
            int(dem_array.shape[1] * resolution_factor)
        )
        dem_array = np.array(Image.fromarray(dem_array).resize(new_shape))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = plt.cm.terrain
    flood_cmap = LinearSegmentedColormap.from_list(
        'flood', ['green', 'yellow', 'red'], N=256
    )
    
    elevation_threshold = np.percentile(dem_array, 30)
    flood_risk = np.where(
        dem_array < elevation_threshold,
        rainfall * 0.1,
        np.nan
    )
    
    danger_threshold = rainfall * 0.05
    danger_mask = flood_risk > danger_threshold
    safe_mask = np.logical_and(dem_array >= elevation_threshold, ~danger_mask)
    
    transform = rasterio.transform.from_bounds(*bounds, dem_array.shape[1], dem_array.shape[0])
    
    danger_shapes = list(features.shapes(danger_mask.astype(np.uint8), transform=transform))
    danger_geoms = [shape(geom) for geom, value in danger_shapes if value == 1]
    danger_gdf = gpd.GeoDataFrame(geometry=danger_geoms, crs="EPSG:4326")
    
    safe_shapes = list(features.shapes(safe_mask.astype(np.uint8), transform=transform))
    safe_geoms = [shape(geom) for geom, value in safe_shapes if value == 1]
    safe_gdf = gpd.GeoDataFrame(geometry=safe_geoms, crs="EPSG:4326")
    
    st.session_state.danger_zone_gdf = danger_gdf
    st.session_state.safe_zone_gdf = safe_gdf
    
    elev_plot = ax.imshow(dem_array, cmap=cmap, alpha=0.7, extent=bounds)
    plt.colorbar(elev_plot, ax=ax, label='Elevation (m)')
    
    risk_plot = ax.imshow(
        flood_risk, 
        cmap=flood_cmap, 
        alpha=0.6,
        vmin=0,
        vmax=rainfall * 0.1,
        extent=bounds
    )
    
    if not danger_gdf.empty:
        danger_gdf.plot(ax=ax, facecolor='red', alpha=0.3, label='Danger Zone')
    if not safe_gdf.empty:
        safe_gdf.plot(ax=ax, facecolor='green', alpha=0.3, label='Safe Zone')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.3, label='Flood Danger Zone'),
        Patch(facecolor='green', alpha=0.3, label='Safe Zone')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(f"Flood Risk Analysis ({rainfall}mm rainfall)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    total_area = dem_array.size
    flood_area = np.sum(flood_risk > 0.5)
    flood_percentage = (flood_area / total_area) * 100
    
    stats = {
        "min_elevation": np.min(dem_array),
        "max_elevation": np.max(dem_array),
        "flood_area": flood_area,
        "flood_percentage": flood_percentage,
        "rainfall_used": rainfall,
        "danger_area": len(danger_geoms),
        "safe_area": len(safe_geoms)
    }
    
    end_performance_monitor(monitor)
    return flood_risk, fig, stats

def buffer_analysis(input_layer, distance):
    """Create buffer zones around features"""
    monitor = start_performance_monitor("buffer_analysis")
    
    input_layer = auto_crs_conversion(input_layer)
    input_layer = validate_geometry(input_layer)
    
    buffered = input_layer.copy()
    buffered.geometry = buffered.geometry.buffer(distance / 111000)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    input_layer.plot(ax=ax, color='blue', alpha=0.5, markersize=50, label='Original')
    buffered.plot(ax=ax, color='red', alpha=0.3, label=f'Buffer ({distance}m)')
    ax.legend()
    ax.set_title("Buffer Analysis")
    
    end_performance_monitor(monitor)
    return buffered, fig

def solar_site_selection(slope_data, substations, max_slope=5, buffer_distance=10):
    """Identify suitable solar farm locations with proper georeferencing"""
    monitor = start_performance_monitor("solar_site_selection")
    
    buffer_deg = buffer_distance / 111
    
    slope_mask = slope_data < max_slope
    
    substation_buffers = substations.copy()
    substation_buffers.geometry = substation_buffers.geometry.buffer(buffer_deg)
    
    # Create suitable areas polygon
    suitable_polygons = []
    for _, row in substation_buffers.iterrows():
        suitable_polygons.append(row.geometry)
    
    if suitable_polygons:
        solar_suitable_areas = gpd.GeoDataFrame(
            geometry=suitable_polygons,
            crs="EPSG:4326"
        )
        st.session_state.solar_suitable_areas = solar_suitable_areas
    else:
        st.session_state.solar_suitable_areas = None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    slope_plot = ax.imshow(slope_data, cmap='viridis', alpha=0.5)
    plt.colorbar(slope_plot, ax=ax, label='Slope (degrees)')
    
    substations.plot(ax=ax, color='red', markersize=50, label='Substations')
    substation_buffers.plot(ax=ax, color='blue', alpha=0.3, label=f'{buffer_distance}km Buffer')
    
    ax.set_title(f"Suitable Solar Farm Locations (Slope < {max_slope}°)")
    ax.legend()
    
    suitable_area = np.sum(slope_mask) * 0.01
    
    stats = {
        "suitable_area": suitable_area,
        "num_substations": len(substations),
        "max_slope": max_slope,
        "buffer_distance": buffer_distance
    }
    
    end_performance_monitor(monitor)
    return fig, stats

def hospital_site_selection(existing_hospitals, roads, min_distance=5):
    """Identify suitable hospital locations with proper georeferencing"""
    monitor = start_performance_monitor("hospital_site_selection")
    
    min_distance_deg = min_distance / 111
    
    exclusion_zones = existing_hospitals.copy()
    exclusion_zones.geometry = exclusion_zones.geometry.buffer(min_distance_deg)
    
    # Create suitable areas (entire area minus exclusion zones)
    # For simplicity, we'll use the convex hull of hospitals as the area
    if not existing_hospitals.empty:
        area_polygon = existing_hospitals.unary_union.convex_hull
        suitable_area = area_polygon.difference(exclusion_zones.unary_union)
        
        hospital_suitable_areas = gpd.GeoDataFrame(
            geometry=[suitable_area],
            crs="EPSG:4326"
        )
        st.session_state.hospital_suitable_areas = hospital_suitable_areas
    else:
        st.session_state.hospital_suitable_areas = None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    existing_hospitals.plot(ax=ax, color='red', markersize=50, label='Existing Hospitals')
    exclusion_zones.plot(ax=ax, color='orange', alpha=0.3, label=f'{min_distance}km Exclusion')
    
    if hasattr(roads, 'nodes'):
        nodes, edges = ox.graph_to_gdfs(roads)
        edges.plot(ax=ax, color='gray', linewidth=1, label='Roads')
    else:
        roads.plot(ax=ax, color='gray', linewidth=1, label='Roads')
    
    ax.set_title(f"Suitable Hospital Locations (> {min_distance}km from existing hospitals)")
    ax.legend()
    
    total_area = 10000
    exclusion_area = len(existing_hospitals) * (min_distance_deg ** 2) * 3.14
    suitable_area = total_area - exclusion_area
    
    stats = {
        "suitable_area": suitable_area,
        "num_existing": len(existing_hospitals),
        "min_distance": min_distance
    }
    
    end_performance_monitor(monitor)
    return fig, stats

def ask_llm_for_clarification(error_message, context, step):
    """Ask LLM how to resolve an error with enhanced geocoding support"""
    if "ambiguous" in error_message.lower() or "please specify" in error_message.lower():
        prompt = f"""
        The user specified an ambiguous location: "{step.get('parameters', {}).get('location', '')}"
        
        Please ask the user to provide the exact location they want to analyze.
        
        Return in JSON format: {{
            "question": "Your question to the user",
            "suggestions": ["list", "of", "common", "locations"]
        }}
        """
    else:
        prompt = f"""
        We encountered an error during geospatial processing: 
        {error_message}
        
        Current step: {step['tool']} with parameters: {step.get('parameters', {})}
        
        Current context: {context}
        
        How should we resolve this issue? Please provide:
        1. A clear question to ask the user for clarification
        2. Suggestions for what the user might respond
        
        Respond in JSON format: 
        {{
            "question": "Your question to the user",
            "suggestions": ["list", "of", "suggested", "answers"]
        }}
        """
    
    try:
        response = ollama.generate(
            model='mistral',
            prompt=prompt,
            format='json',
            options={'temperature': 0.3}
        )
        return json.loads(response['response'])
    except:
        return {
            "question": "How would you like to proceed?",
            "suggestions": ["Skip this step", "Try again", "Provide different parameters"]
        }

def ask_llm_for_parameter_fix(error_message, context, step):
    """Ask LLM to fix parameters based on error"""
    prompt = f"""
    We encountered an error during geospatial processing: 
    {error_message}
    
    Current step: {step['tool']} with parameters: {step.get('parameters', {})}
    
    Current context: {context}
    
    How should we adjust the parameters to resolve this issue? 
    Please return the corrected parameters in JSON format.
    
    Respond in JSON format: 
    {{
        "parameters": {{ "param1": "value1", "param2": "value2" }}
    }}
    """
    
    try:
        response = ollama.generate(
            model='mistral',
            prompt=prompt,
            format='json',
            options={'temperature': 0.2}
        )
        return json.loads(response['response'])
    except:
        return step.get('parameters', {})

def extract_numeric_value(input_str):
    """Extract numeric value from string input"""
    if isinstance(input_str, (int, float)):
        return input_str
    
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(input_str))
    if numbers:
        try:
            return float(numbers[0])
        except:
            pass
    
    return None

def preprocess_query(raw_query):
    """Convert unstructured user query into structured format with enhanced location extraction"""
    prompt = f"""
    As a geospatial analyst, convert this user query into a structured format:
    
    User Query: "{raw_query}"
    
    Important: 
    - Extract the location name as accurately as possible
    - If no location is mentioned, set location to an empty string
    - For ambiguous terms like 'my area', set location to an empty string
    - Identify the task type: solar, flood, hospital, or other
    
    Return JSON format with these fields:
    {{
        "task": "string",  // One of: solar, flood, hospital, other
        "location": "string",  // Extracted location name
        "parameters": {{
            // Key-value pairs of parameters (e.g., "rainfall_mm": 150, "max_slope": 5)
        }},
        "reasoning": "string"  // Your reasoning for this interpretation
    }}
    
    Example Output:
    {{
        "task": "flood",
        "location": "Mumbai",
        "parameters": {{"rainfall_mm": 150}},
        "reasoning": "User mentioned flood risk in Mumbai with 150mm rainfall"
    }}
    """
    
    try:
        response = ollama.generate(
            model='mistral',
            prompt=prompt,
            format='json',
            options={'temperature': 0.1}
        )
        result = json.loads(response['response'])
        
        if 'location' not in result or not result['location'].strip():
            result['location'] = ""
            
        rainfall_match = re.search(r'(\d+)\s*mm', raw_query, re.IGNORECASE)
        if rainfall_match:
            result['parameters']['rainfall_mm'] = float(rainfall_match.group(1))
            
        return result
    except Exception as e:
        st.error(f"Failed to preprocess query: {str(e)}")
        return {
            "task": "other",
            "location": "",
            "parameters": {},
            "reasoning": "Could not interpret query"
        }

def ask_llm_for_query_clarification(raw_query):
    """Simplified query clarification - only ask for location"""
    return {
        "question": "Please specify the exact location you want to analyze:",
        "missing_info": ["location"]
    }

def generate_summary(query, results, task_type):
    """Generate non-technical summary based on analysis results"""
    try:
        location = results.get('location', {}).get('name', 'the area')
        
        if task_type == "Flood Risk":
            stats = results.get('flood_stats', {})
            rainfall = stats.get('rainfall_used', 200)
            risk_percent = stats.get('flood_percentage', 0)
            
            prompt = f"""
            The user asked: "{query}"
            
            We analyzed flood risk for {location} with {rainfall}mm of rainfall.
            Key findings:
            - {risk_percent:.1f}% of the area is at risk of flooding
            - Identified {stats.get('danger_area', 0)} danger zones
            - Identified {stats.get('safe_area', 0)} safe zones
            
            Please write a concise 3-4 sentence summary for non-technical stakeholders:
            1. Explain what this means in simple terms
            2. Highlight key implications
            3. Suggest any precautions
            Use simple language without technical jargon.
            """
            
        elif task_type == "Renewable Energy":
            stats = results.get('solar_stats', {})
            area = stats.get('suitable_area', 0)
            
            prompt = f"""
            The user asked: "{query}"
            
            We analyzed solar farm suitability in {location}.
            Key findings:
            - Found {area:.1f} sq km of suitable land
            - Considered slope under {stats.get('max_slope', 5)} degrees
            - Proximity to {stats.get('num_substations', 0)} substations
            
            Please write a concise 3-4 sentence summary for non-technical stakeholders:
            1. Explain what this means for solar development
            2. Highlight key advantages of the location
            3. Suggest next steps
            Use simple language without technical jargon.
            """
            
        else:  # Urban Planning
            stats = results.get('hospital_stats', {})
            area = stats.get('suitable_area', 0)
            
            prompt = f"""
            The user asked: "{query}"
            
            We analyzed hospital site suitability in {location}.
            Key findings:
            - Found {area:.1f} sq km of suitable land
            - Maintained {stats.get('min_distance', 5)} km from existing hospitals
            - Considered proximity to road network
            
            Please write a concise 3-4 sentence summary for non-technical stakeholders:
            1. Explain what this means for healthcare access
            2. Highlight key advantages of the location
            3. Suggest next steps
            Use simple language without technical jargon.
            """
        
        response = ollama.generate(
            model='mistral',
            prompt=prompt,
            options={'temperature': 0.2}
        )
        return response['response']
    
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}")
        return "Could not generate summary. Please see technical results above."

def execute_step(step):
    """Execute a workflow step with enhanced error handling"""
    tool = step['tool']
    params = step.get('parameters', {})
    
    log_reasoning("Workflow", f"Executing step: {tool} with parameters: {params}")
    st.session_state.execution_history.append({
        "step": tool,
        "parameters": params,
        "timestamp": datetime.now().isoformat()
    })
    
    step_header = st.container()
    with step_header:
        st.markdown(f"### ⚙️ Step: {tool}")
        st.write(f"**Parameters:** `{params}`")
        if "reasoning" in step:
            st.info(f"**Reasoning:** {step['reasoning']}")
    
    try:
        if tool == "geocode":
            location = params.get('location', '')
            if not location:
                log_reasoning("Parameter Handling", "No location specified. Using default: Mumbai")
                location = "Mumbai"
            
            ambiguous_terms = ["myarea", "my area", "current location", "user's location", "near me", "user's area"]
            if location.lower() in ambiguous_terms:
                raise ValueError("Ambiguous location specified - please clarify exact location")
            
            result = geocode(location)
            if result:
                st.session_state.results['location'] = result
                
                with st.expander("📍 Location Map", expanded=True):
                    m = leafmap.Map(center=(result['latitude'], result['longitude']), zoom=10)
                    m.add_marker(location=(result['latitude'], result['longitude']), popup=result['name'])
                    m.to_streamlit(height=400)
                
                st.success(f"**Geocoding Complete:** Found {result['name']} at ({result['latitude']}, {result['longitude']})")
                return True
            else:
                alt_url = "https://nominatim.openstreetmap.org/search"
                alt_params = {"q": location, "format": "json", "countrycodes": "in"}
                alt_response = requests.get(alt_url, params=alt_params, headers={"User-Agent": "GeospatialAI"})
                if alt_response.status_code == 200:
                    alt_data = alt_response.json()
                    if alt_data:
                        st.session_state.alternative_locations = [item['display_name'] for item in alt_data[:5]]
                
                raise ValueError(f"Could not geocode location: {location}")
        
        elif tool == "get_dem":
            if 'location' in st.session_state.results:
                location_info = st.session_state.results['location']
                dem, bounds = get_dem(location_info['name'])
                st.session_state.results['dem'] = dem
                st.session_state.results['dem_bounds'] = bounds
                
                with st.expander("🗻 Digital Elevation Model", expanded=True):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(dem, cmap='terrain', extent=bounds)
                    plt.colorbar(im, ax=ax, label='Elevation (m)')
                    ax.set_title(f"Elevation Model for {location_info['name']}")
                    st.pyplot(fig)
                
                st.success("**DEM Generated:** Synthetic elevation model created")
                return True
            else:
                raise ValueError("Location required to generate DEM")
        
        elif tool == "calculate_slope":
            if 'dem' in st.session_state.results and 'dem_bounds' in st.session_state.results:
                dem = st.session_state.results['dem']
                bounds = st.session_state.results['dem_bounds']
                slope, profile = calculate_slope(dem, bounds)
                st.session_state.results['slope'] = slope
                
                with st.expander("📐 Slope Analysis", expanded=True):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(slope, cmap='viridis', extent=bounds)
                    plt.colorbar(im, ax=ax, label='Slope (degrees)')
                    ax.set_title(f"Terrain Slope for {st.session_state.results['location']['name']}")
                    st.pyplot(fig)
                
                st.success("**Slope Calculated:** Terrain slope derived from DEM")
                return True
            else:
                raise ValueError("DEM required for slope calculation")
        
        elif tool == "get_osm_data":
            if 'location' in st.session_state.results:
                location = st.session_state.results['location']['name']
                feature_type = params.get('feature_type', 'substation')
                gdf = get_osm_data(location, feature_type)
                st.session_state.results[f'osm_{feature_type}'] = gdf
                
                with st.expander(f"🗺️ OSM {feature_type.capitalize()} Data", expanded=True):
                    if feature_type == "road":
                        fig, ax = ox.plot_graph(gdf, show=False, close=False)
                        st.pyplot(fig)
                    else:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        gdf.plot(ax=ax, color='red', markersize=50)
                        ax.set_title(f"{feature_type.capitalize()} in {location}")
                        st.pyplot(fig)
                
                st.success(f"**OSM Data Fetched:** Found {len(gdf)} {feature_type} features")
                return True
            else:
                raise ValueError("Location required to fetch OSM data")
        
        elif tool == "flood_risk_model":
            if 'dem' in st.session_state.results and 'dem_bounds' in st.session_state.results:
                rainfall = params.get('rainfall_mm', 150)
                if 'rainfall_mm' in st.session_state.structured_query.get('parameters', {}):
                    rainfall = st.session_state.structured_query['parameters']['rainfall_mm']
                
                dem = st.session_state.results['dem']
                bounds = st.session_state.results['dem_bounds']
                
                flood_risk, fig, stats = flood_risk_model(dem, rainfall, bounds)
                st.session_state.results['flood_risk'] = flood_risk
                st.session_state.results['flood_stats'] = stats
                
                with st.expander("🌊 Flood Risk Analysis", expanded=True):
                    st.pyplot(fig)
                    st.write(f"**Rainfall used:** {rainfall} mm")
                    st.write(f"**Flood risk area:** {stats['flood_percentage']:.2f}%")
                    st.write(f"**Danger zones:** {stats['danger_area']} areas identified")
                    st.write(f"**Safe zones:** {stats['safe_area']} areas identified")
                    
                    if st.session_state.danger_zone_gdf is not None and not st.session_state.danger_zone_gdf.empty:
                        st.subheader("Interactive Flood Risk Map")
                        m = leafmap.Map(center=(st.session_state.results['location']['latitude'], 
                                        st.session_state.results['location']['longitude']), 
                                      zoom=10)
                        
                        m.add_gdf(st.session_state.danger_zone_gdf, layer_name="Danger Zones", 
                                 fill_colors=['red'], style={'fillOpacity': 0.4})
                        
                        if st.session_state.safe_zone_gdf is not None and not st.session_state.safe_zone_gdf.empty:
                            m.add_gdf(st.session_state.safe_zone_gdf, layer_name="Safe Zones", 
                                     fill_colors=['green'], style={'fillOpacity': 0.4})
                        
                        loc = st.session_state.results['location']
                        m.add_marker(location=(loc['latitude'], loc['longitude']), popup=loc['name'])
                        
                        m.to_streamlit(height=500)
                
                st.success(f"**Flood Risk Model Created:** {stats['flood_percentage']:.2f}% area at risk")
                return True
            else:
                raise ValueError("DEM required for flood risk analysis")
        
        elif tool == "solar_site_selection":
            if 'slope' in st.session_state.results and 'osm_substation' in st.session_state.results:
                slope = st.session_state.results['slope']
                substations = st.session_state.results['osm_substation']
                max_slope = params.get('max_slope', 5)
                buffer_distance = params.get('buffer_distance', 10)
                
                fig, stats = solar_site_selection(slope, substations, max_slope, buffer_distance)
                st.session_state.results['solar_stats'] = stats
                
                with st.expander("☀️ Solar Farm Suitability", expanded=True):
                    st.pyplot(fig)
                    
                    # Create interactive map
                    st.subheader("Interactive Solar Farm Map")
                    m = leafmap.Map(center=(st.session_state.results['location']['latitude'], 
                                  st.session_state.results['location']['longitude']), 
                              zoom=10)
                    
                    # Add substations
                    if not substations.empty:
                        m.add_gdf(substations, layer_name="Substations", marker_type='marker', 
                                 marker_icon='bolt', marker_kwds={'color': 'red', 'prefix': 'fa'})
                    
                    # Add buffer zones
                    buffer_deg = buffer_distance / 111
                    substation_buffers = substations.copy()
                    substation_buffers.geometry = substation_buffers.geometry.buffer(buffer_deg)
                    
                    if not substation_buffers.empty:
                        m.add_gdf(substation_buffers, layer_name="Buffer Zones", 
                                 style={'fillColor': 'blue', 'fillOpacity': 0.3, 'color': 'blue'})
                    
                    # Add suitable areas if available
                    if st.session_state.solar_suitable_areas is not None:
                        m.add_gdf(st.session_state.solar_suitable_areas, layer_name="Suitable Areas", 
                                 style={'fillColor': 'green', 'fillOpacity': 0.4, 'color': 'green'})
                    
                    # Add location marker
                    loc = st.session_state.results['location']
                    m.add_marker(location=(loc['latitude'], loc['longitude']), popup=loc['name'])
                    
                    m.to_streamlit(height=500)
                
                st.success(f"**Solar Sites Identified:** {stats['suitable_area']:.2f} sq km suitable area")
                return True
            else:
                raise ValueError("Slope data and substation locations required")
        
        elif tool == "hospital_site_selection":
            if 'osm_hospital' in st.session_state.results and 'osm_road' in st.session_state.results:
                hospitals = st.session_state.results['osm_hospital']
                roads = st.session_state.results['osm_road']
                min_distance = params.get('min_distance', 5)
                
                fig, stats = hospital_site_selection(hospitals, roads, min_distance)
                st.session_state.results['hospital_stats'] = stats
                
                with st.expander("🏥 Hospital Site Suitability", expanded=True):
                    st.pyplot(fig)
                    
                    # Create interactive map
                    st.subheader("Interactive Hospital Site Map")
                    m = leafmap.Map(center=(st.session_state.results['location']['latitude'], 
                                  st.session_state.results['location']['longitude']), 
                              zoom=10)
                    
                    # Add existing hospitals
                    if not hospitals.empty:
                        m.add_gdf(hospitals, layer_name="Existing Hospitals", marker_type='marker', 
                                 marker_icon='hospital', marker_kwds={'color': 'red', 'prefix': 'fa'})
                    
                    # Add exclusion zones
                    min_distance_deg = min_distance / 111
                    exclusion_zones = hospitals.copy()
                    exclusion_zones.geometry = exclusion_zones.geometry.buffer(min_distance_deg)
                    
                    if not exclusion_zones.empty:
                        m.add_gdf(exclusion_zones, layer_name="Exclusion Zones", 
                                 style={'fillColor': 'orange', 'fillOpacity': 0.3, 'color': 'orange'})
                    
                    # Add roads
                    if isinstance(roads, nx.MultiDiGraph):
                        nodes, edges = ox.graph_to_gdfs(roads)
                        if not edges.empty:
                            m.add_gdf(edges, layer_name="Roads", style={'color': 'gray', 'weight': 2})
                    elif not roads.empty:
                        m.add_gdf(roads, layer_name="Roads", style={'color': 'gray', 'weight': 2})
                    
                    # Add suitable areas if available
                    if st.session_state.hospital_suitable_areas is not None:
                        m.add_gdf(st.session_state.hospital_suitable_areas, layer_name="Suitable Areas", 
                                 style={'fillColor': 'green', 'fillOpacity': 0.4, 'color': 'green'})
                    
                    # Add location marker
                    loc = st.session_state.results['location']
                    m.add_marker(location=(loc['latitude'], loc['longitude']), popup=loc['name'])
                    
                    m.to_streamlit(height=500)
                
                st.success(f"**Hospital Sites Identified:** {stats['suitable_area']:.2f} sq km suitable area")
                return True
            else:
                raise ValueError("Hospital and road data required")
        
        return True
    
    except Exception as e:
        error_msg = str(e)
        log_reasoning("Error", f"Error in {tool}: {error_msg}")
        
        st.session_state.error_context = {
            "tool": tool,
            "params": params,
            "error": error_msg,
            "available_results": list(st.session_state.results.keys())
        }
        st.session_state.clarification_visible = True
        
        return False

# Streamlit UI
st.set_page_config(layout="wide", page_title="🌍 Geospatial Analysis Assistant", page_icon="🌐")
st.title("🌍 Advanced Geospatial Workflow Engine")
st.write("Chain-of-Thought Reasoning for Complex Spatial Analysis")

# Sidebar for system monitoring
with st.sidebar:
    st.header("System Monitor")
    col1, col2 = st.columns(2)
    col1.metric("Memory Usage", f"{psutil.virtual_memory().percent}%")
    col2.metric("CPU Usage", f"{psutil.cpu_percent()}%")
    
    if st.session_state.reasoning_log:
        st.divider()
        st.subheader("Recent Logs")
        for entry in st.session_state.reasoning_log[-3:]:
            st.caption(f"{entry['timestamp']} - {entry['message']}")

# Main content area
col1, col2 = st.columns([3, 1])
with col1:
    # User input
    query = st.text_input("Enter geospatial query:", "Where is flooding likely in Mumbai with 200mm rainfall?")
    
    if st.button("Execute Analysis", type="primary"):
        # Clear previous state
        st.session_state.results = {}
        st.session_state.error_context = {}
        st.session_state.user_clarification = None
        st.session_state.retry_count = 0
        st.session_state.reasoning_log = []
        st.session_state.performance_metrics = {}
        st.session_state.alternative_locations = []
        st.session_state.clarification_visible = False
        st.session_state.query_attempts = 0
        st.session_state.current_step = 0
        st.session_state.execution_history = []
        st.session_state.flood_zones = None
        st.session_state.danger_zone_gdf = None
        st.session_state.safe_zone_gdf = None
        st.session_state.solar_suitable_areas = None
        st.session_state.hospital_suitable_areas = None
        st.session_state.original_query = query
        
        # Preprocess the query with LLM
        with st.spinner("Processing your query..."):
            structured_query = preprocess_query(query)
            st.session_state.structured_query = structured_query
            log_reasoning("Query Processing", f"Structured query: {json.dumps(structured_query, indent=2)}")
            
            # Validate the query
            is_valid, validation_msg = validate_query(structured_query)
            log_reasoning("Query Validation", f"Validation: {is_valid} - {validation_msg}")
            
            if not is_valid:
                st.session_state.query_clarification = {
                    "question": f"❌ {validation_msg}. Please provide missing information:",
                    "missing_info": ["location"] if not structured_query.get('location', '').strip() else []
                }
                st.session_state.query_attempts += 1
            else:
                task_map = {
                    "solar": "Renewable Energy",
                    "flood": "Flood Risk",
                    "hospital": "Urban Planning",
                    "other": "Other"
                }
                st.session_state.task_type = task_map.get(structured_query.get("task", "other"), "Other")
                log_reasoning("Task Classification", f"Classified as: {st.session_state.task_type}")
                st.session_state.query_clarification = None

if st.session_state.query_clarification and st.session_state.query_attempts <= MAX_QUERY_ATTEMPTS:
    with col1:
        st.divider()
        st.subheader("⚠️ Input Required")
        st.write(st.session_state.query_clarification['question'])
        
        user_responses = {}
        col1, col2 = st.columns(2)
        
        # Location field
        if "location" in st.session_state.query_clarification.get('missing_info', []):
            user_responses['location'] = st.text_input("Location name:", key="location_input")
        
        # Rainfall field for flood analysis
        if "rainfall_mm" in st.session_state.query_clarification.get('missing_info', []):
            user_responses['rainfall_mm'] = st.number_input(
                "Rainfall amount (mm):", 
                min_value=1, 
                max_value=1000, 
                value=150,
                key="rainfall_input"
            )
        
        if st.button("Submit Information", key="submit_query_clarification"):
            # Process user responses
            if user_responses:
                new_query = {
                    "task": st.session_state.structured_query.get("task", "other"),
                    "location": user_responses.get('location', st.session_state.structured_query.get('location', '')),
                    "parameters": st.session_state.structured_query.get("parameters", {})
                }
                
                # Add rainfall parameter if provided
                if 'rainfall_mm' in user_responses:
                    new_query['parameters']['rainfall_mm'] = user_responses['rainfall_mm']
                
                new_query['reasoning'] = "User provided missing information through clarification"
                
                st.session_state.structured_query = new_query
                st.session_state.query_attempts = 0
                st.session_state.query_clarification = None
                
                # Re-validate
                is_valid, _ = validate_query(new_query)
                if is_valid:
                    st.rerun()
                else:
                    st.error("Still missing required information. Please try again.")
            else:
                st.warning("Please provide the requested information")

if st.session_state.structured_query and not st.session_state.query_clarification:
    # Final validation check
    is_valid, validation_msg = validate_query(st.session_state.structured_query)
    if not is_valid:
        st.error(f"❌ Unable to proceed: {validation_msg}")
        st.stop()
    
    location = st.session_state.structured_query.get('location', '').strip()
    if not location:
        st.error("Location is still missing. Please restart and provide a location.")
        st.stop()
    
    validated_location = location
    
    if st.session_state.task_type == "Renewable Energy":
        workflow_steps = [
            {"tool": "geocode", "parameters": {"location": validated_location}, "reasoning": "Identify location coordinates"},
            {"tool": "get_dem", "parameters": {}, "reasoning": "Fetch elevation data for slope analysis"},
            {"tool": "calculate_slope", "parameters": {}, "reasoning": "Calculate terrain slope"},
            {"tool": "get_osm_data", "parameters": {"feature_type": "substation"}, "reasoning": "Get power substation locations"},
            {"tool": "solar_site_selection", "parameters": {"max_slope": 5, "buffer_distance": 10}, "reasoning": "Identify suitable solar farm locations"}
        ]
    elif st.session_state.task_type == "Flood Risk":
        workflow_steps = [
            {"tool": "geocode", "parameters": {"location": validated_location}, "reasoning": "Identify location coordinates"},
            {"tool": "get_dem", "parameters": {}, "reasoning": "Fetch elevation data for flood modeling"},
            {"tool": "flood_risk_model", "parameters": {"rainfall_mm": 150}, "reasoning": "Calculate flood risk areas"}
        ]
    else:
        workflow_steps = [
            {"tool": "geocode", "parameters": {"location": validated_location}, "reasoning": "Identify location coordinates"},
            {"tool": "get_osm_data", "parameters": {"feature_type": "hospital"}, "reasoning": "Get existing hospital locations"},
            {"tool": "get_osm_data", "parameters": {"feature_type": "road"}, "reasoning": "Get road network data"},
            {"tool": "hospital_site_selection", "parameters": {"min_distance": 5}, "reasoning": "Identify suitable hospital locations"}
        ]
    
    st.session_state.workflow = workflow_steps
    log_reasoning("Planning", f"Generated workflow: {json.dumps(workflow_steps, indent=2)}")

if st.session_state.workflow:
    with col1:
        st.divider()
        st.subheader("Workflow Execution")
        
        progress = st.session_state.current_step / len(st.session_state.workflow)
        st.progress(progress)
        
        if st.session_state.current_step < len(st.session_state.workflow):
            step = st.session_state.workflow[st.session_state.current_step]
            success = execute_step(step)
            
            if success:
                st.session_state.current_step += 1
                if st.session_state.current_step < len(st.session_state.workflow):
                    st.rerun()
            else:
                if st.session_state.retry_count < MAX_RETRIES:
                    st.warning("Pausing workflow for user clarification...")
                    st.session_state.clarification_visible = True
                else:
                    st.error("Skipping step after maximum retries")
                    st.session_state.current_step += 1
                    st.rerun()
        else:
            st.success("✅ Workflow completed successfully!")
            
            st.divider()
            st.subheader("Final Results")
            
            # Generate and display summary for stakeholders
            summary = generate_summary(
                st.session_state.original_query,
                st.session_state.results,
                st.session_state.task_type
            )
            
            st.subheader("🌍 Summary for Stakeholders")
            st.info(summary)
            
            # Task-specific results display
            if st.session_state.task_type == "Renewable Energy" and 'solar_stats' in st.session_state.results:
                stats = st.session_state.results['solar_stats']
                st.markdown(f"""
                **Solar Farm Site Selection Summary**
                
                - **Suitable Area:** {stats['suitable_area']:.2f} sq km
                - **Max Allowed Slope:** {stats['max_slope']}°
                - **Proximity to Substations:** {stats['buffer_distance']} km
                - **Number of Substations:** {stats['num_substations']}
                
                Suitable areas are shown in the visualization above.
                """)
                
                # Show the interactive map again for consistency
                if 'osm_substation' in st.session_state.results:
                    st.subheader("Solar Farm Site Map")
                    substations = st.session_state.results['osm_substation']
                    buffer_distance = st.session_state.structured_query['parameters'].get('buffer_distance', 10)
                    
                    m = leafmap.Map(center=(st.session_state.results['location']['latitude'], 
                                  st.session_state.results['location']['longitude']), 
                              zoom=10)
                    
                    if not substations.empty:
                        m.add_gdf(substations, layer_name="Substations", marker_type='marker', 
                                 marker_icon='bolt', marker_kwds={'color': 'red', 'prefix': 'fa'})
                    
                    buffer_deg = buffer_distance / 111
                    substation_buffers = substations.copy()
                    substation_buffers.geometry = substation_buffers.geometry.buffer(buffer_deg)
                    
                    if not substation_buffers.empty:
                        m.add_gdf(substation_buffers, layer_name="Buffer Zones", 
                                 style={'fillColor': 'blue', 'fillOpacity': 0.3, 'color': 'blue'})
                    
                    if st.session_state.solar_suitable_areas is not None:
                        m.add_gdf(st.session_state.solar_suitable_areas, layer_name="Suitable Areas", 
                                 style={'fillColor': 'green', 'fillOpacity': 0.4, 'color': 'green'})
                    
                    loc = st.session_state.results['location']
                    m.add_marker(location=(loc['latitude'], loc['longitude']), popup=loc['name'])
                    
                    m.to_streamlit(height=500)
                
            elif st.session_state.task_type == "Flood Risk" and 'flood_stats' in st.session_state.results:
                stats = st.session_state.results['flood_stats']
                st.markdown(f"""
                **Flood Risk Analysis Summary**
                
                - **Rainfall Used:** {stats['rainfall_used']} mm
                - **Area at Risk:** {stats['flood_percentage']:.2f}%
                - **Min Elevation:** {stats['min_elevation']:.2f} m
                - **Max Elevation:** {stats['max_elevation']:.2f} m
                - **Danger Zones:** {stats['danger_area']} areas identified
                - **Safe Zones:** {stats['safe_area']} areas identified
                """)
                
                # Interactive flood map
                st.subheader("Flood Risk Map")
                m = leafmap.Map(center=(st.session_state.results['location']['latitude'], 
                              st.session_state.results['location']['longitude']), 
                          zoom=10)
                
                if st.session_state.danger_zone_gdf is not None and not st.session_state.danger_zone_gdf.empty:
                    m.add_gdf(st.session_state.danger_zone_gdf, layer_name="Danger Zones", 
                             fill_colors=['red'], style={'fillOpacity': 0.4})
                
                if st.session_state.safe_zone_gdf is not None and not st.session_state.safe_zone_gdf.empty:
                    m.add_gdf(st.session_state.safe_zone_gdf, layer_name="Safe Zones", 
                             fill_colors=['green'], style={'fillOpacity': 0.4})
                
                loc = st.session_state.results['location']
                m.add_marker(location=(loc['latitude'], loc['longitude']), popup=loc['name'])
                
                m.to_streamlit(height=500)
                
                # Download button
                if st.session_state.danger_zone_gdf is not None and not st.session_state.danger_zone_gdf.empty:
                    st.download_button(
                        label="📥 Download Danger Zones as GeoJSON",
                        data=st.session_state.danger_zone_gdf.to_json(),
                        file_name="danger_zones.geojson",
                        mime="application/json"
                    )
                
            elif st.session_state.task_type == "Urban Planning" and 'hospital_stats' in st.session_state.results:
                stats = st.session_state.results['hospital_stats']
                st.markdown(f"""
                **Hospital Site Selection Summary**
                
                - **Suitable Area:** {stats['suitable_area']:.2f} sq km
                - **Min Distance from Existing Hospitals:** {stats['min_distance']} km
                - **Number of Existing Hospitals:** {stats['num_existing']}
                """)
                
                # Show the interactive map again for consistency
                if 'osm_hospital' in st.session_state.results and 'osm_road' in st.session_state.results:
                    hospitals = st.session_state.results['osm_hospital']
                    roads = st.session_state.results['osm_road']
                    min_distance = st.session_state.structured_query['parameters'].get('min_distance', 5)
                    
                    st.subheader("Hospital Site Map")
                    m = leafmap.Map(center=(st.session_state.results['location']['latitude'], 
                                  st.session_state.results['location']['longitude']), 
                              zoom=10)
                    
                    if not hospitals.empty:
                        m.add_gdf(hospitals, layer_name="Existing Hospitals", marker_type='marker', 
                                 marker_icon='hospital', marker_kwds={'color': 'red', 'prefix': 'fa'})
                    
                    min_distance_deg = min_distance / 111
                    exclusion_zones = hospitals.copy()
                    exclusion_zones.geometry = exclusion_zones.geometry.buffer(min_distance_deg)
                    
                    if not exclusion_zones.empty:
                        m.add_gdf(exclusion_zones, layer_name="Exclusion Zones", 
                                 style={'fillColor': 'orange', 'fillOpacity': 0.3, 'color': 'orange'})
                    
                    if isinstance(roads, nx.MultiDiGraph):
                        nodes, edges = ox.graph_to_gdfs(roads)
                        if not edges.empty:
                            m.add_gdf(edges, layer_name="Roads", style={'color': 'gray', 'weight': 2})
                    elif not roads.empty:
                        m.add_gdf(roads, layer_name="Roads", style={'color': 'gray', 'weight': 2})
                    
                    if st.session_state.hospital_suitable_areas is not None:
                        m.add_gdf(st.session_state.hospital_suitable_areas, layer_name="Suitable Areas", 
                                 style={'fillColor': 'green', 'fillOpacity': 0.4, 'color': 'green'})
                    
                    loc = st.session_state.results['location']
                    m.add_marker(location=(loc['latitude'], loc['longitude']), popup=loc['name'])
                    
                    m.to_streamlit(height=500)

if st.session_state.clarification_visible and st.session_state.error_context:
    with col1:
        st.divider()
        st.subheader("⚠️ Workflow Needs Clarification")
        
        clarification = ask_llm_for_clarification(
            st.session_state.error_context['error'],
            st.session_state.error_context,
            {"tool": st.session_state.error_context['tool'], "parameters": st.session_state.error_context['params']}
        )
        
        if "ambiguous" in st.session_state.error_context['error'].lower() or "please specify" in st.session_state.error_context['error'].lower():
            st.write("We need more information about the location:")
            st.write(clarification['question'])
            
            common_locations = list(PREDEFINED_LOCATIONS.keys())
            st.write("Common locations:")
            cols = st.columns(3)
            for i, loc in enumerate(common_locations):
                with cols[i % 3]:
                    if st.button(loc, key=f"loc_{loc}"):
                        st.session_state.user_clarification = loc
                        st.session_state.retry_count += 1
                        st.session_state.clarification_visible = False
                        st.rerun()
            
            user_input = st.text_input("Or enter your location:", key="ambiguous_location_input")
            if st.button("Submit Location", key="submit_location"):
                if user_input:
                    st.session_state.user_clarification = user_input
                    st.session_state.retry_count += 1
                    st.session_state.clarification_visible = False
                    st.rerun()
                else:
                    st.warning("Please enter a location name")
                    
        else:
            st.write(clarification['question'])
            
            if clarification.get('suggestions'):
                st.write("Suggestions:")
                cols = st.columns(len(clarification['suggestions']))
                for i, suggestion in enumerate(clarification['suggestions']):
                    with cols[i]:
                        if st.button(suggestion, key=f"suggestion_{i}"):
                            st.session_state.user_clarification = suggestion
                            st.session_state.retry_count += 1
                            st.session_state.clarification_visible = False
                            st.rerun()
            
            user_input = st.text_input("Or enter your response:", key="clarification_input")
            if st.button("Submit Response", key="submit_clarification"):
                st.session_state.user_clarification = user_input
                st.session_state.retry_count += 1
                st.session_state.clarification_visible = False
                st.rerun()

with col2:
    st.subheader("Workflow Steps")
    
    if st.session_state.workflow:
        for i, step in enumerate(st.session_state.workflow):
            status = "✅" if i < st.session_state.current_step else "➡️" if i == st.session_state.current_step else "◻️"
            color = "green" if i < st.session_state.current_step else "blue" if i == st.session_state.current_step else "gray"
            
            with st.expander(f"{status} Step {i+1}: {step['tool']}", expanded=(i == st.session_state.current_step)):
                st.caption(f"**Parameters:** {step.get('parameters', {})}")
                if "reasoning" in step:
                    st.info(step['reasoning'])
    
    if st.session_state.reasoning_log:
        st.divider()
        st.subheader("Reasoning Log")
        
        log_df = pd.DataFrame(st.session_state.reasoning_log)
        st.dataframe(log_df, height=300)

if st.session_state.performance_metrics:
    with col1:
        st.divider()
        st.subheader("Performance Metrics")
        
        metrics_df = pd.DataFrame(st.session_state.performance_metrics).T.reset_index()
        metrics_df.columns = ['Step', 'Runtime (s)', 'Memory Used (bytes)']
        st.dataframe(metrics_df)
        
        total_runtime = sum(metrics['runtime'] for metrics in st.session_state.performance_metrics.values())
        total_memory = sum(metrics['memory_used'] for metrics in st.session_state.performance_metrics.values())
        
        st.write(f"**Total Runtime**: {total_runtime:.2f} seconds")
        st.write(f"**Total Memory Used**: {total_memory / (1024*1024):.2f} MB")

if st.session_state.workflow and st.session_state.current_step == len(st.session_state.workflow):
    with col1:
        st.divider()
        st.download_button(
            label="📥 Export Workflow as JSON",
            data=json.dumps(st.session_state.workflow, indent=2),
            file_name="geospatial_workflow.json",
            mime="application/json"
        )
        
        if st.session_state.task_type == "Renewable Energy" and 'solar_stats' in st.session_state.results:
            st.download_button(
                label="📊 Download Solar Suitability Report",
                data=f"Suitable Area: {st.session_state.results['solar_stats']['suitable_area']:.2f} sq km",
                file_name="solar_suitability_report.txt",
                mime="text/plain"
            )
        elif st.session_state.task_type == "Flood Risk" and 'flood_stats' in st.session_state.results:
            st.download_button(
                label="📊 Download Flood Risk Report",
                data=f"Flood Risk Area: {st.session_state.results['flood_stats']['flood_percentage']:.2f}%",
                file_name="flood_risk_report.txt",
                mime="text/plain"
            )
        elif st.session_state.task_type == "Urban Planning" and 'hospital_stats' in st.session_state.results:
            st.download_button(
                label="📊 Download Hospital Suitability Report",
                data=f"Suitable Area: {st.session_state.results['hospital_stats']['suitable_area']:.2f} sq km",
                file_name="hospital_suitability_report.txt",
                mime="text/plain"
            )

if st.session_state.workflow and st.session_state.current_step == len(st.session_state.workflow):
    with col1:
        if st.button("🔄 Start New Analysis", type="primary"):
            st.session_state.workflow = []
            st.session_state.current_step = 0
            st.rerun()