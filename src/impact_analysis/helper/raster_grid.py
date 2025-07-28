"""
Raster-based grid operations for efficient high-resolution computation.
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box
from src.utils.hurricane_geom import get_nicaragua_boundary


def create_raster_grid(bounds, resolution, crs="EPSG:4326"):
    """
    Create a raster grid efficiently using rasterio.
    
    Args:
        bounds: (minx, miny, maxx, maxy) bounding box
        resolution: grid resolution in degrees
        crs: coordinate reference system
    
    Returns:
        raster_data: numpy array with grid indices
        transform: rasterio transform
        crs: coordinate reference system
    """
    minx, miny, maxx, maxy = bounds
    
    # Calculate grid dimensions
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    
    # Create transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Create grid indices array
    grid_indices = np.arange(width * height).reshape(height, width)
    
    return grid_indices, transform, crs


def raster_to_vector_grid(raster_data, transform, crs, value_column="value"):
    """
    Convert raster grid to vector grid (GeoDataFrame) efficiently.
    
    Args:
        raster_data: numpy array with values
        transform: rasterio transform
        crs: coordinate reference system
        value_column: name for the value column
    
    Returns:
        GeoDataFrame with grid cells and values
    """
    height, width = raster_data.shape
    
    # Generate grid cell geometries efficiently
    geometries = []
    values = []
    
    for row in range(height):
        for col in range(width):
            # Get cell bounds from transform
            x_min, y_min = transform * (col, row)
            x_max, y_max = transform * (col + 1, row + 1)
            
            # Create geometry
            geom = box(x_min, y_min, x_max, y_max)
            geometries.append(geom)
            values.append(raster_data[row, col])
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {value_column: values},
        geometry=geometries,
        crs=crs
    )
    
    return gdf


def compute_vulnerability_raster(schools_gdf, bounds, resolution, crs="EPSG:4326"):
    """
    Compute vulnerability using raster-based operations.
    
    Args:
        schools_gdf: GeoDataFrame with school points
        bounds: (minx, miny, maxx, maxy) bounding box
        resolution: grid resolution in degrees
        crs: coordinate reference system
    
    Returns:
        GeoDataFrame with vulnerability grid
    """
    minx, miny, maxx, maxy = bounds
    
    # Calculate grid dimensions
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    
    # Create transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Initialize vulnerability raster
    vulnerability_raster = np.zeros((height, width), dtype=np.int32)
    
    # Convert schools to raster coordinates
    school_coords = np.array([
        [point.x, point.y] for point in schools_gdf.geometry
    ])
    
    # Convert to raster coordinates
    raster_coords = []
    for coord in school_coords:
        try:
            raster_col, raster_row = ~transform * (coord[0], coord[1])
            raster_coords.append([int(raster_col), int(raster_row)])
        except:
            # Skip invalid coordinates
            continue
    
    raster_coords = np.array(raster_coords)
    
    # Count schools in each cell
    for coord in raster_coords:
        col, row = coord
        if 0 <= row < height and 0 <= col < width:
            vulnerability_raster[row, col] += 1
    
    # Convert to vector grid
    gdf = raster_to_vector_grid(vulnerability_raster, transform, crs, "school_count")
    
    return gdf


def compute_exposure_raster(landslide_file, bounds, resolution, resampling_method="mean", crs="EPSG:4326"):
    """
    Compute exposure using raster-based operations.
    
    Args:
        landslide_file: path to landslide raster file
        bounds: (minx, miny, maxx, maxy) bounding box
        resolution: grid resolution in degrees
        resampling_method: resampling method (mean, min, max)
        crs: coordinate reference system
    
    Returns:
        GeoDataFrame with exposure grid
    """
    minx, miny, maxx, maxy = bounds
    
    # Calculate grid dimensions
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    
    # Create transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Read landslide raster
    with rasterio.open(landslide_file) as src:
        # Get the full raster data
        data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        
        # Create a simple grid-based approach instead of complex windowing
        # This is more robust for the initial implementation
        exposure_raster = np.zeros((height, width), dtype=np.float32)
        
        # For each grid cell, sample the raster
        for row in range(height):
            for col in range(width):
                # Get cell center coordinates
                x_center = minx + (col + 0.5) * resolution
                y_center = miny + (row + 0.5) * resolution
                
                # Convert to raster coordinates
                try:
                    raster_col, raster_row = ~src_transform * (x_center, y_center)
                    raster_col, raster_row = int(raster_col), int(raster_row)
                    
                    # Check bounds
                    if 0 <= raster_row < data.shape[0] and 0 <= raster_col < data.shape[1]:
                        value = data[raster_row, raster_col]
                        # Handle nodata values (negative values or -9999)
                        if value < 0 or value == -9999:
                            exposure_raster[row, col] = 0.0
                        else:
                            exposure_raster[row, col] = value
                    else:
                        exposure_raster[row, col] = 0.0
                except:
                    exposure_raster[row, col] = 0.0
    
    # Convert to vector grid
    gdf = raster_to_vector_grid(exposure_raster, transform, crs, "probability")
    
    return gdf


def get_nicaragua_bounds():
    """Get Nicaragua bounding box."""
    nicaragua_gdf = get_nicaragua_boundary()
    return nicaragua_gdf.total_bounds


def compute_population_raster(population_files, bounds, resolution, crs="EPSG:4326"):
    """
    Compute population using raster-based operations with clipping.
    
    Args:
        population_files: list of paths to population raster files
        bounds: (minx, miny, maxx, maxy) bounding box for clipping
        resolution: grid resolution in degrees
        crs: coordinate reference system
    
    Returns:
        GeoDataFrame with population grid
    """
    minx, miny, maxx, maxy = bounds
    
    # Calculate grid dimensions
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    
    # Create transform for output grid
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Initialize combined population raster
    combined_population = np.zeros((height, width), dtype=np.float32)
    
    # Process each population file
    for file_path in population_files:
        with rasterio.open(file_path) as src:
            # Clip the raster to our bounds
            window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
            
            # Read the clipped data
            try:
                data = src.read(1, window=window)
                
                # Handle nodata values
                data = np.where(data <= -99999, 0, data)
                
                # Resample to our target resolution if needed
                if data.shape != (height, width):
                    # Use proper raster resampling that preserves total population
                    # Calculate the ratio of original to target resolution
                    orig_height, orig_width = data.shape
                    
                    # Calculate how many original cells map to each target cell
                    height_ratio = orig_height / height
                    width_ratio = orig_width / width
                    
                    # Resample by summing the original cells that fall into each target cell
                    resampled_data = np.zeros((height, width), dtype=np.float32)
                    
                    for row in range(height):
                        for col in range(width):
                            # Calculate the range of original cells that map to this target cell
                            start_row = int(row * height_ratio)
                            end_row = int((row + 1) * height_ratio)
                            start_col = int(col * width_ratio)
                            end_col = int((col + 1) * width_ratio)
                            
                            # Sum the population in this range
                            cell_population = data[start_row:end_row, start_col:end_col].sum()
                            resampled_data[row, col] = cell_population
                    
                    data = resampled_data
                
                # Add to combined population
                combined_population += data
                
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
                continue
    
    # Convert to vector grid
    gdf = raster_to_vector_grid(combined_population, transform, crs, "population_count")
    
    return gdf 