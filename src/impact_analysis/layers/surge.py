"""
Storm surge layer for impact analysis.

This module provides the SurgeLayer class that integrates storm surge
calculations with the impact analysis framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
import geopandas as gpd
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import warnings
from shapely.geometry import Point

from src.impact_analysis.layers.base import ExposureLayer
from src.data_prep.surge_helper.surge_processor import SurgeProcessor
from src.utils.config_utils import load_config, get_config_value
from src.utils.path_utils import get_data_path, ensure_directory
from src.utils.hurricane_geom import get_nicaragua_boundary
from src.impact_analysis.utils.plotting_utils import plot_layer_with_scales

warnings.filterwarnings('ignore')


class SurgeLayer(ExposureLayer):
    """Storm surge layer for impact analysis."""

    def __init__(self, hurricane_file: str = None, bathymetry_file: str = None, config: dict = None, cache_dir: str = None, use_cache: bool = True):
        """Initialize the surge layer."""
        if config is None:
            config = load_config("config/surge_config.yaml")
            
        super().__init__(config)
        
        self.processor = SurgeProcessor()
        self.hurricane_file = hurricane_file
        self.bathymetry_file = bathymetry_file
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # Surge state variables
        self.eta = None  # Water level (m)
        self.u = None  # U-velocity (m/s)
        self.v = None  # V-velocity (m/s)
        self.max_eta = None  # Maximum water level envelope
        
        # Grid data
        self.grid_lats = None
        self.grid_lons = None
        self.grid_depths = None
        self.dx = None
        self.dy = None
        
        # Hurricane data
        self.hurricane_data = None
        self.track_history = {}  # Store track history for motion calculations
        
        # Results storage
        self.results = {}
        
        # Auto-initialize if hurricane file is provided
        if hurricane_file is not None:
            print("Auto-initializing surge layer...")
            if self.load_data():
                print("Checking for cached surge results...")
                if self._load_cached_results():
                    print("Using cached surge results")
                else:
                    print("Running ensemble simulation...")
                    self.results = self.run_ensemble_simulation()
                    if self.results:
                        print(f"Surge ensemble simulation completed with {len(self.results)} members")
                        # Save high-resolution data to preprocessed folder
                        self.save_high_resolution_data()
                        # Cache the results
                        self._save_cached_results()
                    else:
                        print("Warning: No surge results generated")
            else:
                print("Warning: Failed to load surge data")

    def load_data(self, hurricane_file: str = None, bathymetry_file: Optional[str] = None) -> bool:
        """Load hurricane and bathymetry data."""
        print(f"Loading surge data...")
        
        # Use stored hurricane_file if not provided
        if hurricane_file is None:
            hurricane_file = self.hurricane_file
        if bathymetry_file is None:
            bathymetry_file = self.bathymetry_file
            
        # Load hurricane data
        if not self._load_hurricane_data(hurricane_file):
            return False
            
        # Process bathymetry
        if not self.processor.process_bathymetry(bathymetry_file):
            return False
            
        # Get processed grid data
        self.grid_lats, self.grid_lons, self.grid_depths, self.dx, self.dy = self.processor.get_grid_data()
        
        # Initialize surge state variables
        self.eta = np.zeros_like(self.grid_lats, dtype=np.float32)
        self.u = np.zeros_like(self.grid_lats, dtype=np.float32)
        self.v = np.zeros_like(self.grid_lats, dtype=np.float32)
        self.max_eta = np.zeros_like(self.grid_lats, dtype=np.float32)
        
        print("Surge data loaded successfully!")
        return True

    def _load_hurricane_data(self, hurricane_file: str) -> bool:
        """Load and preprocess hurricane ensemble data."""
        print(f"Loading hurricane data: {hurricane_file}")

        if not Path(hurricane_file).exists():
            print(f"Hurricane file not found: {hurricane_file}")
            return False

        try:
            self.hurricane_data = pd.read_csv(hurricane_file)
            self.hurricane_data['valid_time'] = pd.to_datetime(self.hurricane_data['valid_time'])

            print(f"Loaded {len(self.hurricane_data)} hurricane records")
            print(f"  Ensemble members: {self.hurricane_data['ensemble_member'].nunique()}")
            print(f"  Time range: {self.hurricane_data['valid_time'].min()} to {self.hurricane_data['valid_time'].max()}")

            # Validate required columns
            required_cols = ['latitude', 'longitude', 'wind_speed', 'pressure', 'ensemble_member']
            missing_cols = [col for col in required_cols if col not in self.hurricane_data.columns]
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                return False

            # Check for wind radii availability
            wind_radii_cols = [col for col in self.hurricane_data.columns if 'radius_' in col and 'knot_winds_' in col]
            rmw_col = 'radius_of_maximum_winds_km'

            if wind_radii_cols:
                print(f"  Found {len(wind_radii_cols)} wind radii columns - will use observed asymmetric structure")
            else:
                print(f"  No wind radii found - will use symmetric Holland model")

            if rmw_col in self.hurricane_data.columns:
                print(f"  Found RMW column - will use observed values when available")
            else:
                print(f"  No RMW column - will calculate empirically")

            # Verify storm intensity
            max_wind = self.hurricane_data['wind_speed'].max()
            min_pressure = self.hurricane_data['pressure'].min()
            print(f"  Storm intensity: {max_wind:.0f}kt peak, {min_pressure:.0f}mb minimum")

            return True
            
        except Exception as e:
            print(f"Error loading hurricane data: {e}")
            return False

    def calculate_storm_motion(self, storm_data, member, step):
        """Calculate storm forward speed and direction from track history."""
        if member not in self.track_history:
            self.track_history[member] = []

        # Store current position
        current_pos = {
            'lat': storm_data['latitude'],
            'lon': storm_data['longitude'],
            'time': storm_data.get('valid_time', step)
        }
        self.track_history[member].append(current_pos)

        # Need at least 2 points to calculate motion
        if len(self.track_history[member]) < 2:
            # Default values for first point
            return 6.0, 270.0  # 6 m/s westward

        # Get previous position
        prev_pos = self.track_history[member][-2]

        # Calculate distance and time difference
        dlat = current_pos['lat'] - prev_pos['lat']
        dlon = current_pos['lon'] - prev_pos['lon']

        # Convert to meters
        dlat_m = dlat * 111000  # meters per degree latitude
        dlon_m = dlon * 111000 * np.cos(np.radians(current_pos['lat']))

        distance_m = np.sqrt(dlat_m ** 2 + dlon_m ** 2)

        # Time difference (assume 6-hour intervals if no time data)
        if hasattr(current_pos['time'], 'total_seconds'):
            dt_hours = (current_pos['time'] - prev_pos['time']).total_seconds() / 3600
        else:
            dt_hours = 6.0  # Default 6-hour interval

        if dt_hours <= 0:
            dt_hours = 6.0

        # Calculate speed and direction
        speed_ms = distance_m / (dt_hours * 3600)  # m/s
        direction_deg = np.arctan2(dlon_m, dlat_m) * 180 / np.pi
        direction_deg = (direction_deg + 360) % 360  # 0-360°

        # Limit to reasonable values
        speed_ms = np.clip(speed_ms, 0, 20)  # 0-40 kt range

        return speed_ms, direction_deg

    def calculate_radius_max_winds(self, storm_data, max_wind_ms, central_pressure, storm_lat):
        """Calculate radius of maximum winds using observed data when available, otherwise empirical relationships."""
        # Use observed RMW when available
        if 'radius_of_maximum_winds_km' in storm_data and not pd.isna(storm_data['radius_of_maximum_winds_km']):
            rmw_observed = storm_data['radius_of_maximum_winds_km']
            return np.clip(rmw_observed, 5, 200)  # Wider range for observed data
        else:
            # Fallback to empirical RMW formula (Willoughby et al. 2006)
            max_wind_kt = max_wind_ms / 0.514
            if central_pressure < 1013:
                rmw = 46.4 * np.exp(-0.0155 * max_wind_kt + 0.0169 * abs(storm_lat))
            else:
                rmw = 25.0  # Default for weaker systems
            return np.clip(rmw, 15, 80)  # Original bounds for calculated RMW

    def calculate_holland_b_parameter(self, max_wind_ms, central_pressure, storm_lat):
        """Calculate Holland B parameter."""
        max_wind_kt = max_wind_ms / 0.514
        pressure_deficit = 1013.25 - central_pressure

        if pressure_deficit > 5:
            holland_b = 1.881 - 0.00557 * max_wind_kt - 0.010917 * abs(storm_lat)
        else:
            holland_b = 1.5

        return np.clip(holland_b, 0.8, 2.5)

    def get_quadrant_from_bearing(self, bearing_rad):
        """Determine quadrant (0=NE, 1=SE, 2=SW, 3=NW) from bearing in radians."""
        bearing_deg = np.degrees(bearing_rad) % 360
        quadrant = np.where(bearing_deg < 90, 0,  # NE
                            np.where(bearing_deg < 180, 1,  # SE
                                     np.where(bearing_deg < 270, 2,  # SW
                                              3)))  # NW
        return quadrant

    def extract_wind_radii(self, storm_data):
        """Extract wind radii data."""
        valid_count = 0
        wind_radii = {}

        for wind_speed in [34, 50, 64]:
            wind_radii[wind_speed] = {}
            for quadrant in ['ne', 'se', 'sw', 'nw']:
                col_name = f'radius_{wind_speed}_knot_winds_{quadrant}_km'

                if col_name in storm_data:
                    value = storm_data[col_name]
                    if not pd.isna(value) and value > 0:
                        wind_radii[wind_speed][quadrant] = float(value)
                        valid_count += 1
                    else:
                        wind_radii[wind_speed][quadrant] = 0
                else:
                    wind_radii[wind_speed][quadrant] = 0

        return wind_radii if valid_count >= 4 else None

    def interpolate_wind_from_radii(self, distance_km, wind_radii, quadrant, max_wind_speed):
        """Interpolate wind speed based on observed wind radii."""
        if wind_radii is None:
            return None

        # Map quadrant numbers to strings
        quadrant_names = ['ne', 'se', 'sw', 'nw']

        # Handle both array and scalar inputs
        if np.isscalar(quadrant):
            quad_name = quadrant_names[quadrant]
            quad_radii = {ws: wind_radii[ws][quad_name] for ws in [34, 50, 64]}
        else:
            # For array inputs, we need to handle each point
            result = np.zeros_like(distance_km)
            for i, quad_idx in enumerate(np.ravel(quadrant)):
                quad_name = quadrant_names[quad_idx]
                quad_radii = {ws: wind_radii[ws][quad_name] for ws in [34, 50, 64]}
                result.ravel()[i] = self._interpolate_single_point(np.ravel(distance_km)[i], quad_radii, max_wind_speed)
            return result

        return self._interpolate_single_point(distance_km, quad_radii, max_wind_speed)

    def _interpolate_single_point(self, distance_km, quad_radii, max_wind_speed):
        """Interpolate wind speed for a single point."""
        # Get radii for this quadrant (convert kt to m/s)
        r34 = quad_radii[34]  # 34 kt = ~17.5 m/s
        r50 = quad_radii[50]  # 50 kt = ~25.7 m/s
        r64 = quad_radii[64]  # 64 kt = ~32.9 m/s

        # More flexible handling of missing radii
        # Find the largest valid radius
        valid_radii = [(r, wind_ms) for r, wind_ms in [(r64, 32.9), (r50, 25.7), (r34, 17.5)] if r > 0]

        if not valid_radii:
            return 0  # No valid radii

        # Sort by radius (largest first)
        valid_radii.sort(key=lambda x: x[0], reverse=True)

        # Estimate RMW from available data
        if r64 > 0:
            rmw_approx = max(r64 * 0.4, 15)  # RMW typically 40% of 64kt radius
        elif r50 > 0:
            rmw_approx = max(r50 * 0.5, 15)  # RMW typically 50% of 50kt radius
        elif r34 > 0:
            rmw_approx = max(r34 * 0.6, 15)  # RMW typically 60% of 34kt radius
        else:
            rmw_approx = 25  # Default

        wind_speed_ms = max_wind_speed  # Start with max wind

        if distance_km < rmw_approx:
            # Inside RMW - scale to max wind
            wind_speed_ms = max_wind_speed * (distance_km / rmw_approx)
        else:
            # Outside RMW - use piecewise linear interpolation with available radii
            for i, (radius, wind_thresh) in enumerate(valid_radii):
                if distance_km <= radius:
                    if i == 0:
                        # Between RMW and first radius
                        wind_speed_ms = max_wind_speed - (max_wind_speed - wind_thresh) * (distance_km - rmw_approx) / (
                                    radius - rmw_approx)
                    else:
                        # Between two radii
                        prev_radius, prev_wind = valid_radii[i - 1]
                        wind_speed_ms = prev_wind - (prev_wind - wind_thresh) * (distance_km - prev_radius) / (
                                    radius - prev_radius)
                    break
            else:
                # Outside all radii - exponential decay
                outermost_radius, outermost_wind = valid_radii[-1]
                wind_speed_ms = outermost_wind * np.exp(-(distance_km - outermost_radius) / 50)

        return max(wind_speed_ms, 0)

    def generate_realistic_hurricane_wind_field(self, storm_data, member, step):
        """
        Generate realistic asymmetric hurricane wind field.
        Uses observed wind radii when available, otherwise Holland model.
        """
        storm_lat = storm_data['latitude']
        storm_lon = storm_data['longitude']
        central_pressure = storm_data['pressure']
        max_wind_speed = storm_data['wind_speed'] * 0.514  # Convert kt to m/s

        # Calculate storm motion
        storm_speed, storm_direction = self.calculate_storm_motion(storm_data, member, step)

        # Distance and bearing from storm center
        dlat = self.grid_lats - storm_lat
        dlon = (self.grid_lons - storm_lon) * np.cos(np.radians(storm_lat))
        distance_km = np.sqrt(dlat ** 2 + dlon ** 2) * 111.0

        # Bearing from storm center (0° = north, 90° = east)
        bearing_rad = np.arctan2(dlon, dlat)

        # Try to use observed wind radii first
        wind_radii = self.extract_wind_radii(storm_data)

        # Calculate wind field
        if wind_radii is not None:
            # Use observed asymmetric wind structure
            quadrant = self.get_quadrant_from_bearing(bearing_rad)
            wind_speed_asymmetric = self.interpolate_wind_from_radii(distance_km, wind_radii, quadrant, max_wind_speed)

            # If interpolation failed, fall back to Holland model
            if wind_speed_asymmetric is None:
                wind_speed_asymmetric = self._generate_holland_wind_field(storm_data, distance_km, bearing_rad,
                                                                          max_wind_speed, central_pressure, storm_lat,
                                                                          storm_speed, storm_direction)
        else:
            # Use Holland model when no wind radii available
            wind_speed_asymmetric = self._generate_holland_wind_field(storm_data, distance_km, bearing_rad,
                                                                      max_wind_speed, central_pressure, storm_lat,
                                                                      storm_speed, storm_direction)

        # Calculate inflow angle (creates spiral structure)
        rmw = self.calculate_radius_max_winds(storm_data, max_wind_speed, central_pressure, storm_lat)
        normalized_radius = distance_km / rmw
        inflow_angle_deg = 15 + 30 * (1 - np.exp(-normalized_radius / 1.2))
        inflow_angle_deg = np.minimum(inflow_angle_deg, 40)

        # Convert to u,v components with spiral structure
        effective_bearing_rad = bearing_rad + np.radians(inflow_angle_deg)
        u_wind = wind_speed_asymmetric * np.sin(effective_bearing_rad)
        v_wind = wind_speed_asymmetric * np.cos(effective_bearing_rad)

        # Surface effects
        land_mask = self.grid_depths >= 0
        water_mask = self.grid_depths < 0

        u_wind[land_mask] *= 0.8  # Reduction over land
        v_wind[land_mask] *= 0.8
        u_wind[water_mask] *= 1.05  # Enhancement over water
        v_wind[water_mask] *= 1.05

        # Pressure field
        pressure_field = np.full_like(self.grid_lats, central_pressure)
        pressure_deficit = 1013.25 - central_pressure
        pressure_variation = pressure_deficit * np.exp(-distance_km / (rmw * 3))
        pressure_field = central_pressure + pressure_variation

        # Final wind field scaling to match observed max_wind_speed
        max_field_speed = np.max(np.sqrt(u_wind ** 2 + v_wind ** 2))
        if max_field_speed > 0:
            scaling_factor = max_wind_speed / max_field_speed
            u_wind *= scaling_factor
            v_wind *= scaling_factor

        return u_wind, v_wind, pressure_field

    def _generate_holland_wind_field(self, storm_data, distance_km, bearing_rad, max_wind_speed, central_pressure,
                                     storm_lat, storm_speed, storm_direction):
        """Generate Holland wind field (fallback when no observed radii)."""
        # Holland wind field parameters
        rmw = self.calculate_radius_max_winds(storm_data, max_wind_speed, central_pressure, storm_lat)
        holland_b = self.calculate_holland_b_parameter(max_wind_speed, central_pressure, storm_lat)

        # Holland profile
        pressure_deficit = 1013.25 - central_pressure

        # Avoid division by zero at storm center
        distance_km_safe = np.maximum(distance_km, 1.0)

        # Holland wind speed formula
        if pressure_deficit > 5:
            try:
                # Holland formula with proper scaling
                holland_term = (holland_b * pressure_deficit * 100) / 1.225  # Convert mb to Pa
                r_ratio = rmw / distance_km_safe
                wind_profile = r_ratio ** holland_b * np.exp(-r_ratio ** holland_b)
                wind_speed_symmetric = np.sqrt(holland_term * wind_profile)

                # Ensure minimum realistic intensity for major hurricanes
                if max_wind_speed > 50:  # Hurricane strength
                    intensity_floor = max_wind_speed * 0.3 * np.exp(-distance_km / 150)
                    wind_speed_symmetric = np.maximum(wind_speed_symmetric, intensity_floor)

                # Cap at reasonable maximum
                wind_speed_symmetric = np.minimum(wind_speed_symmetric, max_wind_speed * 1.5)

            except:
                # Fallback
                wind_speed_symmetric = max_wind_speed * np.exp(-distance_km / (rmw * 0.6))
        else:
            # Weak system
            wind_speed_symmetric = max_wind_speed * np.exp(-distance_km / (rmw * 0.7))

        # Minimum wind field strength
        min_wind_field = max_wind_speed * 0.2 * np.exp(-distance_km / 250)
        wind_speed_symmetric = np.maximum(wind_speed_symmetric, min_wind_field)

        # Asymmetric effects
        # 1. Motion asymmetry
        storm_direction_rad = np.radians(storm_direction)
        relative_bearing = bearing_rad - storm_direction_rad
        motion_asymmetry = 1.0 + 0.3 * (storm_speed / 10.0) * np.cos(relative_bearing - np.pi / 2)
        motion_asymmetry = np.clip(motion_asymmetry, 0.4, 1.4)

        # 2. Coriolis asymmetry
        coriolis_effect = 1.0 + 0.15 * np.cos(bearing_rad - np.pi / 4)
        coriolis_effect = np.clip(coriolis_effect, 0.6, 1.6)

        # 3. Distance-based asymmetry
        distance_asymmetry = 1.0 + 0.15 * np.exp(-distance_km / 80) * np.cos(relative_bearing)
        distance_asymmetry = np.clip(distance_asymmetry, 0.7, 1.5)

        # Apply all asymmetries
        wind_speed_asymmetric = wind_speed_symmetric * motion_asymmetry * coriolis_effect * distance_asymmetry

        return wind_speed_asymmetric

    def calculate_wind_stress(self, u_wind, v_wind):
        """Calculate wind stress using drag coefficient for hurricane conditions."""
        wind_speed = np.sqrt(u_wind ** 2 + v_wind ** 2)

        # Drag coefficient for hurricane conditions
        cd = np.where(wind_speed <= 7,
                      1.2e-3,  # Light winds
                      np.where(wind_speed <= 15,
                               (1.0 + 0.055 * wind_speed) * 1e-3,  # Moderate winds
                               np.where(wind_speed <= 35,
                                        (1.5 + 0.035 * wind_speed) * 1e-3,  # Hurricane winds
                                        np.where(wind_speed <= 60,
                                                 (2.2 + 0.025 * wind_speed) * 1e-3,  # Major hurricane
                                                 3.5e-3))))  # Extreme winds

        # Wind stress components
        base_stress_factor = 1.4

        # Additional boost for hurricane conditions
        max_wind = np.max(wind_speed)
        if max_wind > 50:  # Major hurricane boost
            hurricane_stress_boost = 1.4
        elif max_wind > 35:  # Hurricane boost
            hurricane_stress_boost = 1.2
        else:  # Normal conditions
            hurricane_stress_boost = 1.0

        total_stress_factor = base_stress_factor * hurricane_stress_boost

        air_density = get_config_value(self.config, "surge.air_density", 1.225)
        tau_x = air_density * cd * wind_speed * u_wind * total_stress_factor
        tau_y = air_density * cd * wind_speed * v_wind * total_stress_factor

        return tau_x, tau_y

    def calculate_pressure_gradient(self, pressure_field):
        """Calculate pressure gradient forces."""
        # Convert to Pa and calculate gradients
        pressure_pa = pressure_field * 100

        # Finite difference gradients
        dp_dx = np.gradient(pressure_pa, axis=1) / self.dx
        dp_dy = np.gradient(pressure_pa, axis=0) / self.dy

        # Pressure gradient acceleration (m/s²)
        water_density = get_config_value(self.config, "surge.water_density", 1025.0)
        acc_x = -dp_dx / water_density
        acc_y = -dp_dy / water_density

        return acc_x, acc_y

    def calculate_bottom_friction(self):
        """Calculate bottom friction using Manning's formula."""
        # Total water depth
        total_depth = np.maximum(-self.grid_depths + self.eta, 0.1)

        # Manning friction coefficient
        manning_coef = get_config_value(self.config, "surge.manning_coefficient", 0.025)
        gravity = get_config_value(self.config, "surge.gravity", 9.81)
        velocity_mag = np.sqrt(self.u ** 2 + self.v ** 2)
        manning_cf = (gravity * manning_coef ** 2) / (total_depth ** (1 / 3))

        # Friction forces (acceleration)
        friction_x = manning_cf * velocity_mag * self.u
        friction_y = manning_cf * velocity_mag * self.v

        return friction_x, friction_y

    def surge_physics(self, u_wind, v_wind, pressure_field, dt):
        """
        Surge calculation with shallow water equations and coastal coverage.
        """
        water_mask = (self.grid_depths < 0)

        if not np.any(water_mask):
            return True

        wind_speed = np.sqrt(u_wind ** 2 + v_wind ** 2)
        max_wind = np.max(wind_speed)

        # Calculate forces
        wind_stress_x, wind_stress_y = self.calculate_wind_stress(u_wind, v_wind)
        pressure_acc_x, pressure_acc_y = self.calculate_pressure_gradient(pressure_field)
        friction_x, friction_y = self.calculate_bottom_friction()

        # Pressure effects
        pressure_deficit = 1013.25 - pressure_field

        # Static pressure setup
        static_setup = pressure_deficit * 0.0101

        # Dynamic pressure
        dynamic_setup = np.where(wind_speed > 15,
                                 wind_speed ** 2 * 0.00004,
                                 0)

        # Total water depth
        total_depth = np.maximum(-self.grid_depths + self.eta, 0.5)

        # Hurricane boost
        hurricane_boost = 1.0
        if max_wind > 30:
            hurricane_boost = 1.3 + (max_wind - 30) * 0.008
            hurricane_boost = min(hurricane_boost, 2.0)

        water_density = get_config_value(self.config, "surge.water_density", 1025.0)
        wind_acc_x = wind_stress_x / (water_density * total_depth) * hurricane_boost
        wind_acc_y = wind_stress_y / (water_density * total_depth) * hurricane_boost

        # Friction reduction
        friction_reduction = 0.6 if max_wind > 50 else 0.8
        friction_x *= friction_reduction
        friction_y *= friction_reduction

        # Water level gradients
        deta_dx = np.gradient(self.eta, axis=1) / self.dx
        deta_dy = np.gradient(self.eta, axis=0) / self.dy

        # Update velocities
        gravity = get_config_value(self.config, "surge.gravity", 9.81)
        du_dt = (-gravity * deta_dx + wind_acc_x + pressure_acc_x - friction_x)
        dv_dt = (-gravity * deta_dy + wind_acc_y + pressure_acc_y - friction_y)

        self.u[water_mask] += du_dt[water_mask] * dt
        self.v[water_mask] += dv_dt[water_mask] * dt

        # Continuity equation
        hu = total_depth * self.u
        hv = total_depth * self.v

        dhu_dx = np.gradient(hu, axis=1) / self.dx
        dhv_dy = np.gradient(hv, axis=0) / self.dy

        deta_dt = -(dhu_dx + dhv_dy)

        # Apply water level changes
        surge_increment = deta_dt * dt + static_setup * dt / 2400 + dynamic_setup * dt / 2400
        self.eta[water_mask] += surge_increment[water_mask]

        # Bathymetric effects for coastal coverage
        very_shallow = (self.grid_depths < 0) & (self.grid_depths > -10) & water_mask
        shallow = (self.grid_depths <= -10) & (self.grid_depths > -30) & water_mask
        moderate = (self.grid_depths <= -30) & (self.grid_depths > -50) & water_mask

        # Amplification for broader, realistic coverage
        if max_wind > 50:
            if np.any(very_shallow):
                self.eta[very_shallow] *= 1.4
            if np.any(shallow):
                self.eta[shallow] *= 1.25
            if np.any(moderate):
                self.eta[moderate] *= 1.15
        else:
            if np.any(very_shallow):
                self.eta[very_shallow] *= 1.2
            if np.any(shallow):
                self.eta[shallow] *= 1.1

        # Storm-dependent effects for broader coverage
        if max_wind > 30:
            moderate_wind_mask = wind_speed > (max_wind * 0.4)
            if np.any(moderate_wind_mask & water_mask):
                if max_wind > 60:
                    broad_enhancement = 1.0 + (max_wind - 30) * 0.008
                    broad_enhancement = min(broad_enhancement, 1.4)
                else:
                    broad_enhancement = 1.0 + (max_wind - 30) * 0.006
                    broad_enhancement = min(broad_enhancement, 1.25)

                self.eta[moderate_wind_mask & water_mask] *= broad_enhancement

        # Coastal propagation - apply to all coastal areas
        if max_wind > 40:
            # Apply to all coastal areas (both Caribbean and Pacific)
            coastal_mask = (self.grid_depths < 0) & (self.grid_depths > -20)
            if np.any(coastal_mask):
                from scipy.ndimage import gaussian_filter
                coastal_surge = np.where(coastal_mask, self.eta, 0)
                smoothed_coastal = gaussian_filter(coastal_surge, sigma=1)  # Reduced smoothing

                coastal_boost = 0.2 if max_wind > 60 else 0.1  # Reduced boost
                self.eta[coastal_mask] += smoothed_coastal[coastal_mask] * coastal_boost

        # Damping for surge retention
        damping_factor = 0.9997 if max_wind > 60 else 0.9995
        self.eta *= damping_factor

        # Realistic bounds
        if max_wind > 60:
            max_expected = min(max_wind * 0.06, 6.0)
        elif max_wind > 35:
            max_expected = min(max_wind * 0.05, 4.0)
        else:
            max_expected = min(max_wind * 0.04, 2.0)

        self.eta = np.clip(self.eta, -0.5, max_expected)

        # Handle land areas and inland penetration
        land_mask = self.grid_depths >= 0
        
        # Check if inland penetration is enabled
        inland_penetration_enabled = get_config_value(self.config, "surge.inland_penetration_enabled", False)
        
        if inland_penetration_enabled:
            # Get inland penetration parameters
            max_inland_distance = get_config_value(self.config, "surge.max_inland_distance_km", 5.0)
            inland_decay = get_config_value(self.config, "surge.inland_decay_factor", 0.8)
            coastal_elevation_threshold = get_config_value(self.config, "surge.coastal_elevation_threshold", 5.0)
            
            # Calculate distance from coastline
            distance_from_coast = self._calculate_distance_from_coastline()
            
            # Apply inland penetration for low-lying coastal areas
            inland_mask = (self.grid_depths >= 0) & (distance_from_coast <= max_inland_distance)
            
            if np.any(inland_mask):
                # Get coastal surge values (from adjacent water cells)
                coastal_surge_values = np.where(self.grid_depths < 0, self.eta, 0)
                
                # Find maximum coastal surge near each inland point
                from scipy.ndimage import maximum_filter
                max_coastal_surge = maximum_filter(coastal_surge_values, size=3)
                
                # Apply inland penetration with distance decay
                for i in range(len(self.grid_lats)):
                    for j in range(len(self.grid_lons)):
                        if inland_mask[i, j]:
                            distance_km = distance_from_coast[i, j]
                            if distance_km <= max_inland_distance:
                                # Get maximum coastal surge from nearby water cells
                                nearby_coastal_surge = max_coastal_surge[i, j]
                                
                                if nearby_coastal_surge > 0:
                                    # Apply distance-based decay
                                    decay_factor = inland_decay ** distance_km
                                    inland_surge = nearby_coastal_surge * decay_factor
                                    
                                    # Apply elevation-based reduction
                                    elevation_factor = max(0.1, 1.0 - (self.grid_depths[i, j] / coastal_elevation_threshold))
                                    inland_surge *= elevation_factor
                                    
                                    self.eta[i, j] = inland_surge
                                else:
                                    self.eta[i, j] = 0
                            else:
                                self.eta[i, j] = 0
            else:
                # No inland penetration - set land areas to zero
                self.eta[land_mask] = 0
                self.u[land_mask] = 0
                self.v[land_mask] = 0
        else:
            # Standard behavior - no inland penetration
            self.eta[land_mask] = 0
            self.u[land_mask] = 0
            self.v[land_mask] = 0

        # Update maximum envelope
        self.max_eta = np.maximum(self.max_eta, self.eta)

        return True

    def _calculate_distance_from_coastline(self) -> np.ndarray:
        """Calculate distance from coastline for inland penetration."""
        from scipy.ndimage import distance_transform_edt
        
        # Create coastline mask (where depth transitions from negative to positive)
        coastline_mask = np.zeros_like(self.grid_depths, dtype=bool)
        
        # Find coastline points (where depth is close to 0)
        coastline_threshold = 0.5  # meters
        coastline_mask = np.abs(self.grid_depths) <= coastline_threshold
        
        # Calculate distance from coastline
        distance_from_coast = distance_transform_edt(~coastline_mask)
        
        # Convert from grid units to kilometers
        # Assuming grid resolution is in degrees, convert to km
        grid_resolution_km = get_config_value(self.config, "surge.grid_resolution_km", 3.0)
        distance_from_coast_km = distance_from_coast * grid_resolution_km
        
        return distance_from_coast_km

    def run_ensemble_simulation(self, n_members: Optional[int] = None, max_hours: Optional[int] = None) -> Dict:
        """Run ensemble simulation with physics and export individual results."""
        if n_members is None:
            n_members = get_config_value(self.config, "surge.default_n_members", 10)
        if max_hours is None:
            max_hours = get_config_value(self.config, "surge.max_hours", 48)
            
        print(f"Running ensemble simulation")

        # Get available ensemble members
        available_members = sorted(self.hurricane_data['ensemble_member'].unique())

        # Determine how many members to actually run
        if n_members is None:
            members_to_run = available_members
            print(f"Using all available members: {len(members_to_run)}")
        else:
            members_to_run = available_members[:n_members]
            print(f"Using {len(members_to_run)} of {len(available_members)} available members")

        print(f"Members: {members_to_run}, Duration: {max_hours}h")
        results = {}

        time_step_hours = get_config_value(self.config, "surge.time_step_hours", 0.002)
        dt_seconds = time_step_hours * 3600

        for i, member in enumerate(members_to_run):
            print(f"[{i + 1}/{len(members_to_run)}] Member {member}")

            # Clear track history for this member
            self.track_history = {}

            member_data = self.hurricane_data[self.hurricane_data['ensemble_member'] == member]
            member_data = member_data.sort_values('valid_time')

            if len(member_data) < 5:
                print(f"  Insufficient data ({len(member_data)} points)")
                continue

            max_wind_kt = member_data['wind_speed'].max()
            min_pressure = member_data['pressure'].min()

            print(f"  Storm: {max_wind_kt:.0f}kt, {min_pressure:.0f}mb")

            # Reset state
            self.eta = np.zeros_like(self.grid_lats)
            self.u = np.zeros_like(self.grid_lats)
            self.v = np.zeros_like(self.grid_lats)
            member_max = np.zeros_like(self.grid_lats)

            # Simulation parameters
            max_steps = min(int(max_hours * 3600 / dt_seconds), len(member_data))

            success = True
            max_wind_used = 0.0
            max_surge_achieved = 0.0

            print(f"    Running {max_steps} time steps...")

            for step in range(max_steps):
                try:
                    storm_data = member_data.iloc[step]

                    # Generate wind field
                    u_wind, v_wind, pressure_field = self.generate_realistic_hurricane_wind_field(
                        storm_data, member, step
                    )

                    current_max_wind = np.max(np.sqrt(u_wind ** 2 + v_wind ** 2))
                    max_wind_used = max(max_wind_used, current_max_wind)

                    # Run physics time step
                    if not self.surge_physics(u_wind, v_wind, pressure_field, dt_seconds):
                        print(f"    Failed at step {step}")
                        success = False
                        break

                    current_surge = np.max(self.eta)
                    max_surge_achieved = max(max_surge_achieved, current_surge)
                    member_max = np.maximum(member_max, self.eta)

                    # Report every 6 hours
                    if step % (21600 // dt_seconds) == 0 and step > 0:
                        surge_cells = np.sum(self.eta > 0.05)
                        print(f"    {step * dt_seconds / 3600:.0f}h: Surge={current_surge:.3f}m, "
                              f"Wind={current_max_wind:.1f}m/s, Cells={surge_cells}")

                except Exception as e:
                    print(f"    Error at step {step}: {e}")
                    success = False
                    break

            if success:
                final_max_surge = np.max(member_max)
                water_mask = (self.grid_depths < 0)

                # Calculate surge statistics
                surge_10cm_area = np.sum((member_max > 0.1) & water_mask)
                surge_50cm_area = np.sum((member_max > 0.5) & water_mask)
                surge_100cm_area = np.sum((member_max > 1.0) & water_mask)

                total_water = np.sum(water_mask)
                coverage_percent = surge_10cm_area / total_water * 100 if total_water > 0 else 0

                results[member] = {
                    'max_surge': final_max_surge,
                    'surge_10cm_area': surge_10cm_area,
                    'surge_50cm_area': surge_50cm_area,
                    'surge_100cm_area': surge_100cm_area,
                    'max_wind': max_wind_used,
                    'storm_max_wind': max_wind_kt,
                    'storm_min_pressure': min_pressure,
                    'final_surge_field': member_max.copy(),
                    'coverage_percent': coverage_percent
                }

                print(f"    FINAL: Max={final_max_surge:.3f}m, Coverage={coverage_percent:.1f}%, "
                      f"Wind={max_wind_used:.1f}m/s")

            else:
                print(f"    FAILED")

        return results

    def get_surge_exposure(self, results: Dict) -> np.ndarray:
        """Get surge exposure field from ensemble results."""
        if not results:
            return np.zeros_like(self.grid_lats)
            
        # Calculate ensemble mean surge
        all_surge_fields = [r['final_surge_field'] for r in results.values()]
        mean_surge = np.mean(all_surge_fields, axis=0)
        
        return mean_surge
        
    def get_surge_probability(self, threshold: float = 0.5) -> np.ndarray:
        """Get surge probability map for given threshold."""
        if not hasattr(self, 'results') or not self.results:
            return np.zeros_like(self.grid_lats) if self.grid_lats is not None else np.array([])
            
        all_surge_fields = [r['final_surge_field'] for r in self.results.values()]
        return np.mean([field > threshold for field in all_surge_fields], axis=0)

    def get_plot_metadata(self) -> Dict[str, Any]:
        """Return metadata for plotting this exposure layer."""
        return {
            "layer_type": "exposure",
            "hazard_type": "Surge",
            "data_column": "probability",
            "colormap": "Blues",
            "title_template": "Probability of Storm Surge",
            "legend_template": "Surge Probability per Cell",
            "filename_template": "surge_exposure_{parameters}",
            "special_features": ["axis_limits"]
        }
        
    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        """Plot the exposure layer using universal plotting function."""
        # Use universal plotting function
        plot_layer_with_scales(self, output_dir=output_dir)
        
    def compute_grid(self) -> gpd.GeoDataFrame:
        """Compute the surge exposure grid using standard grid system."""
        try:
            # Get interpolated surge data with proper coastal buffer
            interpolated_surge = self._get_interpolated_surge_data()
            
            if interpolated_surge is None:
                print("Warning: Could not get interpolated surge data")
                # Return empty GeoDataFrame
                from shapely.geometry import Point
                return gpd.GeoDataFrame({'probability': [0.0]}, geometry=[Point(0, 0)], crs="EPSG:4326")
            
            # Use the same grid system as vulnerability layers (same as flood layer)
            from src.utils.hurricane_geom import get_nicaragua_boundary
            from src.utils.config_utils import get_config_value
            from shapely.geometry import box
            
            nicaragua_gdf = get_nicaragua_boundary()
            minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
            
            # Use the same grid resolution as other layers
            if self.config is None:
                grid_res = 0.1  # Default fallback
            else:
                grid_res = get_config_value(self.config, "impact_analysis.grid.resolution_degrees")
                if grid_res is None:
                    grid_res = 0.1  # Fallback if config value not found
            
            # Create grid using standard approach (same as flood layer)
            print(f"DEBUG: Creating grid with bounds: {minx}, {miny}, {maxx}, {maxy}")
            print(f"DEBUG: Grid resolution: {grid_res}")
            print(f"DEBUG: Config is None: {self.config is None}")
            
            grid_cells = []
            x_coords = np.arange(minx, maxx, grid_res)
            y_coords = np.arange(miny, maxy, grid_res)
            
            print(f"DEBUG: x_coords range: {x_coords[0]} to {x_coords[-1]}, {len(x_coords)} points")
            print(f"DEBUG: y_coords range: {y_coords[0]} to {y_coords[-1]}, {len(y_coords)} points")
            
            for x in x_coords:
                for y in y_coords:
                    grid_cells.append(box(x, y, x + grid_res, y + grid_res))
            
            # Create GeoDataFrame with standard grid
            grid_gdf = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:4326")
            
            # Map surge data to grid cells
            probabilities = []
            for cell in grid_gdf.geometry:
                # Find corresponding surge value from interpolated data
                # Map grid coordinates to interpolated surge array indices
                comp_minx, comp_maxx = -89.93, -73.83
                comp_miny, comp_maxy = 7.78, 18.97
                
                # Get cell center
                center = cell.centroid
                x, y = center.x, center.y
                
                # Calculate indices in interpolated surge array
                i = int((x - comp_minx) / (comp_maxx - comp_minx) * 50)
                j = int((y - comp_miny) / (comp_maxy - comp_miny) * 44)
                
                # Clamp indices to valid range
                i = max(0, min(i, 49))
                j = max(0, min(j, 43))
                
                # Get surge value
                surge_value = interpolated_surge[j, i]
                probabilities.append(surge_value)
            
            # Add probability column
            grid_gdf["probability"] = probabilities
            
            print(f"Surge exposure grid: {len(grid_gdf)} cells")
            print(f"Exposure cells > 0: {np.sum(grid_gdf['probability'] > 0)}")
            print(f"Exposure probability range: {np.min(grid_gdf['probability']):.3f} to {np.max(grid_gdf['probability']):.3f}")
            
            return grid_gdf
            
        except Exception as e:
            print(f"Error computing surge grid: {e}")
            # Return empty GeoDataFrame
            from shapely.geometry import Point
            return gpd.GeoDataFrame({'probability': [0.0]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        
    def get_exposure_data(self):
        """Get exposure data for impact analysis framework."""
        # Load data if not already loaded
        if self.grid_depths is None:
            if not self.load_data():
                return None
                
        # Return the surge exposure field
        if hasattr(self, 'results') and self.results:
            return self.get_surge_exposure(self.results)
        else:
            # Return empty exposure if no results
            return np.zeros_like(self.grid_lats) if self.grid_lats is not None else np.array([])

    def save_high_resolution_data(self, output_dir="data/preprocessed/surge/"):
        """Save high-resolution surge data to preprocessed folder."""
        if not hasattr(self, 'results') or not self.results:
            print("No surge results available for saving")
            return
            
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get surge exposure data
        surge_exposure = self.get_surge_exposure(self.results)
        
        # Save as numpy array
        surge_file = output_path / "nicaragua_surge_ensemble_mean.npy"
        np.save(surge_file, surge_exposure)
        
        # Save surge height map as GeoTIFF
        if self.grid_lons is not None and self.grid_lats is not None:
            try:
                import rasterio
                from rasterio.transform import from_bounds
                
                # Calculate bounds from grid coordinates
                min_lon, max_lon = np.min(self.grid_lons), np.max(self.grid_lons)
                min_lat, max_lat = np.min(self.grid_lats), np.max(self.grid_lats)
                
                # Create transform
                transform = from_bounds(min_lon, min_lat, max_lon, max_lat, 
                                     self.grid_lons.shape[1], self.grid_lats.shape[0])
                
                # Save ensemble mean surge as GeoTIFF
                surge_tiff_file = output_path / "nicaragua_surge_ensemble_mean.tif"
                with rasterio.open(
                    surge_tiff_file, 'w',
                    driver='GTiff',
                    height=surge_exposure.shape[0],
                    width=surge_exposure.shape[1],
                    count=1,
                    dtype=surge_exposure.dtype,
                    crs='EPSG:4326',
                    transform=transform
                ) as dst:
                    # Flip the array vertically to correct orientation
                    dst.write(np.flipud(surge_exposure), 1)
                
                # Save individual ensemble members as separate files
                for member_id, result in self.results.items():
                    member_surge = result['final_surge_field']
                    member_tiff_file = output_path / f"nicaragua_surge_member_{member_id}.tif"
                    
                    with rasterio.open(
                        member_tiff_file, 'w',
                        driver='GTiff',
                        height=member_surge.shape[0],
                        width=member_surge.shape[1],
                        count=1,
                        dtype=member_surge.dtype,
                        crs='EPSG:4326',
                        transform=transform
                    ) as dst:
                        # Flip the array vertically to correct orientation
                        dst.write(np.flipud(member_surge), 1)
                
                print(f"  Surge height maps: {surge_tiff_file}")
                print(f"  Individual members: {len(self.results)} files")
                
            except ImportError:
                print("  Note: rasterio not available - skipping GeoTIFF export")
            except Exception as e:
                print(f"  Warning: Could not save GeoTIFF files: {e}")
        
        # Save metadata
        metadata = {
            'grid_shape': surge_exposure.shape,
            'domain_bounds': self.processor.get_domain_bounds(),
            'ensemble_members': len(self.results),
            'max_surge': float(np.max(surge_exposure)),
            'mean_surge': float(np.mean(surge_exposure)),
            'grid_lons': self.grid_lons.tolist() if self.grid_lons is not None else None,
            'grid_lats': self.grid_lats.tolist() if self.grid_lats is not None else None,
            'grid_depths': self.grid_depths.tolist() if self.grid_depths is not None else None
        }
        
        import json
        metadata_file = output_path / "nicaragua_surge_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"High-resolution surge data saved to: {output_path}")
        print(f"  Surge data: {surge_file}")
        print(f"  Metadata: {metadata_file}")
        
        # Also save surge exposure visualization (raw height data)
        self._save_surge_exposure_visualization(output_path, surge_exposure)
        
        # Also save interpolated height data (after inland penetration)
        self._save_interpolated_surge_data(output_path)
        
        # Also create mean and max surge visualizations
        self._create_mean_max_visualizations(output_path)
        
        return str(output_path)

    def _save_surge_exposure_visualization(self, output_path: Path, surge_exposure: np.ndarray):
        """Save surge exposure visualization showing raw surge height data."""
        try:
            # Save statistics only (avoid memory issues with plotting)
            stats_file = output_path / "surge_exposure_statistics.txt"
            with open(stats_file, 'w') as f:
                f.write("Storm Surge Exposure Statistics\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Grid shape: {surge_exposure.shape}\n")
                f.write(f"Maximum surge height: {np.max(surge_exposure):.3f} m\n")
                f.write(f"Mean surge height: {np.mean(surge_exposure):.3f} m\n")
                f.write(f"Standard deviation: {np.std(surge_exposure):.3f} m\n")
                f.write(f"Cells with surge > 0.1m: {np.sum(surge_exposure > 0.1)}\n")
                f.write(f"Cells with surge > 0.5m: {np.sum(surge_exposure > 0.5)}\n")
                f.write(f"Cells with surge > 1.0m: {np.sum(surge_exposure > 1.0)}\n")
                f.write(f"Cells with surge > 2.0m: {np.sum(surge_exposure > 2.0)}\n")
                f.write(f"Cells with surge > 3.0m: {np.sum(surge_exposure > 3.0)}\n")
            
            print(f"  Surge exposure statistics: {stats_file}")
            
            # Also save a simple text summary
            summary_file = output_path / "surge_exposure_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Surge exposure summary:\n")
                f.write(f"- Max height: {np.max(surge_exposure):.3f} m\n")
                f.write(f"- Mean height: {np.mean(surge_exposure):.3f} m\n")
                f.write(f"- Total cells with surge: {np.sum(surge_exposure > 0)}\n")
                f.write(f"- High surge cells (>1m): {np.sum(surge_exposure > 1.0)}\n")
            
            print(f"  Surge exposure summary: {summary_file}")
                
        except Exception as e:
            print(f"  Warning: Could not create surge exposure statistics: {e}")

    def _save_interpolated_surge_data(self, output_path: Path):
        """Save interpolated surge height data (after inland penetration is applied)."""
        try:
            # Get the interpolated surge data that's used for impact analysis
            interpolated_surge = self._get_interpolated_surge_data()
            
            if interpolated_surge is not None:
                # Save interpolated surge data
                interpolated_file = output_path / "nicaragua_surge_interpolated_heights.npy"
                np.save(interpolated_file, interpolated_surge)
                
                # Create visualization of interpolated surge heights
                self._create_interpolated_surge_visualization(output_path, interpolated_surge)
                
                # Save statistics
                stats_file = output_path / "surge_interpolated_statistics.txt"
                with open(stats_file, 'w') as f:
                    f.write("Interpolated Surge Height Statistics (After Inland Penetration)\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Grid shape: {interpolated_surge.shape}\n")
                    f.write(f"Maximum surge height: {np.max(interpolated_surge):.3f} m\n")
                    f.write(f"Mean surge height: {np.mean(interpolated_surge):.3f} m\n")
                    f.write(f"Standard deviation: {np.std(interpolated_surge):.3f} m\n")
                    f.write(f"Cells with surge > 0: {np.sum(interpolated_surge > 0)}\n")
                    f.write(f"Cells with surge > 0.1m: {np.sum(interpolated_surge > 0.1)}\n")
                    f.write(f"Cells with surge > 0.5m: {np.sum(interpolated_surge > 0.5)}\n")
                    f.write(f"Cells with surge > 1.0m: {np.sum(interpolated_surge > 1.0)}\n")
                    f.write(f"Cells with surge > 2.0m: {np.sum(interpolated_surge > 2.0)}\n")
                
                print(f"  Interpolated surge data: {interpolated_file}")
                print(f"  Interpolated surge statistics: {stats_file}")
                
        except Exception as e:
            print(f"  Warning: Could not save interpolated surge data: {e}")

    def _get_interpolated_surge_data(self):
        """Get interpolated surge data using simple extension from original surge points."""
        # Load the ensemble mean surge data
        surge_file = Path("data/preprocessed/surge/nicaragua_surge_ensemble_mean.npy")
        if not surge_file.exists():
            print("Surge ensemble mean file not found")
            return None
        
        surge_data = np.load(surge_file)
        
        # Create computational grid coordinates
        comp_minx, comp_maxx = -89.93, -73.83
        comp_miny, comp_maxy = 7.78, 18.97
        
        comp_lons = np.linspace(comp_minx, comp_maxx, 349)
        comp_lats = np.linspace(comp_miny, comp_maxy, 249)
        
        comp_lon_mesh, comp_lat_mesh = np.meshgrid(comp_lons, comp_lats)
        
        # Get Nicaragua bounds for target grid
        nicaragua_gdf = get_nicaragua_boundary()
        minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
        
        resolution = 0.1
        lons = np.arange(minx, maxx + resolution, resolution)
        lats = np.arange(miny, maxy + resolution, resolution)
        
        # Create target grid points
        grid_lons = []
        grid_lats = []
        for j in range(len(lats) - 1):  # latitude (rows)
            for i in range(len(lons) - 1):  # longitude (columns)
                grid_lons.append((lons[i] + lons[i+1]) / 2)
                grid_lats.append((lats[j] + lats[j+1]) / 2)
        
        # Find significant surge points (threshold approach)
        surge_threshold = get_config_value(self.config, "surge.surge_threshold_m", 0.5)
        valid_surge_mask = surge_data > surge_threshold
        comp_lons_surge = comp_lon_mesh[valid_surge_mask]
        comp_lats_surge = comp_lat_mesh[valid_surge_mask]
        comp_surge_surge = surge_data[valid_surge_mask]
        
        print(f"Significant surge points (> {surge_threshold}m): {len(comp_lons_surge)}")
        
        # Simple surge extension approach
        max_extension_distance_km = get_config_value(self.config, "surge.max_inland_distance_km", 5.0)
        extended_surge = np.zeros(len(grid_lons))
        
        print(f"Applying simple surge extension:")
        print(f"  Max extension distance: {max_extension_distance_km}km")
        
        # For each target grid point, find the nearest surge point and extend if within range
        for i, (lon, lat) in enumerate(zip(grid_lons, grid_lats)):
            # Find the nearest surge point
            if len(comp_lons_surge) > 0:
                distances = np.sqrt((lon - comp_lons_surge)**2 + (lat - comp_lats_surge)**2) * 111.0  # Convert to km
                nearest_idx = np.argmin(distances)
                nearest_distance_km = distances[nearest_idx]
                nearest_surge = comp_surge_surge[nearest_idx]
                
                # Only extend if within the extension distance
                if nearest_distance_km <= max_extension_distance_km:
                    # Apply distance-based decay
                    decay_factor = 1.0 - (nearest_distance_km / max_extension_distance_km)
                    extended_surge[i] = nearest_surge * decay_factor
            else:
                extended_surge[i] = 0.0
        
        print(f"Extended surge range: {np.min(extended_surge):.3f} to {np.max(extended_surge):.3f}")
        print(f"Extended cells > 0: {np.sum(extended_surge > 0)}")
        print(f"Extended cells > 0.1: {np.sum(extended_surge > 0.1)}")
        
        # Reshape to grid
        reshaped_surge = extended_surge.reshape(len(lats)-1, len(lons)-1)
        print(f"Reshaped surge shape: {reshaped_surge.shape}")
        
        return reshaped_surge

    def _create_interpolated_surge_visualization(self, output_path: Path, interpolated_surge: np.ndarray):
        """Create visualization of interpolated surge heights."""
        try:
            import matplotlib.pyplot as plt
            from src.utils.hurricane_geom import get_nicaragua_boundary
            
            # Debug: Print interpolated surge statistics
            print(f"DEBUG VIS: Interpolated surge shape: {interpolated_surge.shape}")
            print(f"DEBUG VIS: Interpolated surge range: {np.min(interpolated_surge):.3f} to {np.max(interpolated_surge):.3f}")
            print(f"DEBUG VIS: Cells with surge > 0: {np.sum(interpolated_surge > 0)}")
            print(f"DEBUG VIS: Cells with surge > 0.1: {np.sum(interpolated_surge > 0.1)}")
            print(f"DEBUG VIS: Cells with surge > 0.5: {np.sum(interpolated_surge > 0.5)}")
            
            # Use the EXACT same grid coordinates as the interpolation
            resolution = get_config_value(self.config, "impact_analysis.grid.resolution_degrees", 0.1)
            nicaragua_gdf = get_nicaragua_boundary()
            minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
            
            print(f"DEBUG VIS: Grid bounds: {minx:.3f} to {maxx:.3f} W, {miny:.3f} to {maxy:.3f} N")
            
            # Create grid coordinates (EXACT same as interpolation)
            lons = np.arange(minx, maxx + resolution, resolution)
            lats = np.arange(miny, maxy + resolution, resolution)
            
            print(f"DEBUG VIS: Grid shape: {len(lons)-1} x {len(lats)-1}")
            print(f"DEBUG VIS: Expected interpolated shape: {len(lats)-1} x {len(lons)-1}")
            print(f"DEBUG VIS: Actual interpolated shape: {interpolated_surge.shape}")
            
            # Create meshgrid for plotting (EXACT same as interpolation)
            # The meshgrid should match the coordinate order used in interpolation
            lon_mesh, lat_mesh = np.meshgrid(lons[:-1], lats[:-1])
            
            print(f"DEBUG VIS: Meshgrid shapes - lon: {lon_mesh.shape}, lat: {lat_mesh.shape}")
            print(f"DEBUG VIS: Meshgrid coordinate ranges:")
            print(f"  Lon mesh: {np.min(lon_mesh):.3f} to {np.max(lon_mesh):.3f}")
            print(f"  Lat mesh: {np.min(lat_mesh):.3f} to {np.max(lat_mesh):.3f}")
            
            # Verify the shapes match
            if interpolated_surge.shape != (len(lats)-1, len(lons)-1):
                print(f"ERROR: Shape mismatch! Interpolated: {interpolated_surge.shape}, Expected: {(len(lats)-1, len(lons)-1)}")
                return
            
            # Debug: Find where the surge values are located
            surge_indices = np.where(interpolated_surge > 0.1)
            if len(surge_indices[0]) > 0:
                print(f"DEBUG VIS: High surge locations (lat, lon):")
                for i in range(min(10, len(surge_indices[0]))):  # Show first 10
                    lat_idx, lon_idx = surge_indices[0][i], surge_indices[1][i]
                    lat_val = lats[lat_idx]
                    lon_val = lons[lon_idx]
                    surge_val = interpolated_surge[lat_idx, lon_idx]
                    print(f"  ({lat_val:.3f}, {lon_val:.3f}): {surge_val:.3f}m")
            
            # Debug: Check if the issue is with the coordinate mapping
            print(f"DEBUG VIS: Grid coordinate ranges:")
            print(f"  Lons: {lons[0]:.3f} to {lons[-2]:.3f} ({len(lons)-1} points)")
            print(f"  Lats: {lats[0]:.3f} to {lats[-2]:.3f} ({len(lats)-1} points)")
            
            # Debug: Check a few specific grid cells
            print(f"DEBUG VIS: Sample grid cell values:")
            for i in range(0, min(5, len(lats)-1), 10):
                for j in range(0, min(5, len(lons)-1), 10):
                    if interpolated_surge[i, j] > 0:
                        print(f"  Grid[{i},{j}] ({lats[i]:.3f}, {lons[j]:.3f}): {interpolated_surge[i, j]:.3f}m")
            
            # Debug: Check if the high surge locations from earlier are actually in the grid
            print(f"DEBUG VIS: Verifying high surge locations in grid:")
            for lat_val, lon_val in [(10.913, -86.686), (11.013, -83.386), (14.213, -85.886)]:
                # Find closest grid cell
                lat_idx = np.argmin(np.abs(lats - lat_val))
                lon_idx = np.argmin(np.abs(lons - lon_val))
                grid_val = interpolated_surge[lat_idx, lon_idx]
                print(f"  ({lat_val:.3f}, {lon_val:.3f}) -> Grid[{lat_idx},{lon_idx}] ({lats[lat_idx]:.3f}, {lons[lon_idx]:.3f}): {grid_val:.3f}m")
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Linear scale
            im1 = ax1.pcolormesh(lon_mesh, lat_mesh, interpolated_surge, 
                                 cmap='Blues', shading='auto', alpha=0.8)
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Surge Height (m)', fontsize=12)
            
            # Add Nicaragua boundary
            nicaragua_gdf.plot(ax=ax1, edgecolor='black', facecolor='none', linewidth=2)
            ax1.set_title('Interpolated Surge Heights - Linear Scale', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Longitude', fontsize=12)
            ax1.set_ylabel('Latitude', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Log scale (for better visibility of small values)
            # Add small value to avoid log(0)
            log_surge = np.log10(interpolated_surge + 0.01)
            im2 = ax2.pcolormesh(lon_mesh, lat_mesh, log_surge, 
                                 cmap='Blues', shading='auto', alpha=0.8)
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Log10(Surge Height + 0.01) (m)', fontsize=12)
            
            # Add Nicaragua boundary
            nicaragua_gdf.plot(ax=ax2, edgecolor='black', facecolor='none', linewidth=2)
            ax2.set_title('Interpolated Surge Heights - Log Scale', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Longitude', fontsize=12)
            ax2.set_ylabel('Latitude', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Set consistent bounds
            for ax in [ax1, ax2]:
                ax.set_xlim(-87.5, -82.5)
                ax.set_ylim(10.5, 15.5)
            
            # Save the plot
            plot_file = output_path / "surge_interpolated_heights_visualization.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Interpolated surge visualization: {plot_file}")
            
        except Exception as e:
            print(f"  Warning: Could not create interpolated surge visualization: {e}")

    def _create_mean_max_visualizations(self, output_dir: Path):
        """Create mean and max surge visualizations for final results."""
        try:
            # Load the ensemble mean surge data for mean visualization
            mean_surge = self._get_interpolated_surge_data()
            if mean_surge is None:
                print("Warning: Could not get interpolated surge data for mean/max visualizations")
                return
            
            # Load individual ensemble members for max visualization
            max_surge = self._get_max_surge_data()
            if max_surge is None:
                print("Warning: Could not get max surge data, using mean for both")
                max_surge = mean_surge
            
            # Get Nicaragua bounds for grid
            nicaragua_gdf = get_nicaragua_boundary()
            minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
            
            resolution = 0.1
            lons = np.arange(minx, maxx + resolution, resolution)
            lats = np.arange(miny, maxy + resolution, resolution)
            
            # Create grid coordinates for visualization
            lon_centers = [(lons[i] + lons[i+1]) / 2 for i in range(len(lons) - 1)]
            lat_centers = [(lats[j] + lats[j+1]) / 2 for j in range(len(lats) - 1)]
            lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
            
            # Create results directory
            results_dir = Path("data/results/impact_analysis/surge_population_ensemble")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Create separate figure for MEAN surge
            fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
            im1 = ax1.pcolormesh(lon_mesh, lat_mesh, mean_surge, 
                                cmap='Blues', shading='auto', alpha=0.8)
            plt.colorbar(im1, ax=ax1, label='Mean Surge Height (m)')
            nicaragua_gdf.plot(ax=ax1, edgecolor='black', facecolor='none', linewidth=2)
            ax1.set_title('Mean Surge Heights (Ensemble Average)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Longitude', fontsize=12)
            ax1.set_ylabel('Latitude', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            mean_file = results_dir / "surge_exposure__mean.png"
            plt.savefig(mean_file, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # Create separate figure for MAX surge
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
            im2 = ax2.pcolormesh(lon_mesh, lat_mesh, max_surge, 
                                cmap='Reds', shading='auto', alpha=0.8)
            plt.colorbar(im2, ax=ax2, label='Max Surge Height (m)')
            nicaragua_gdf.plot(ax=ax2, edgecolor='black', facecolor='none', linewidth=2)
            ax2.set_title('Max Surge Heights (Ensemble Maximum)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Longitude', fontsize=12)
            ax2.set_ylabel('Latitude', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            max_file = results_dir / "surge_exposure__max.png"
            plt.savefig(max_file, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            print(f"  Mean surge visualization: {mean_file}")
            print(f"  Max surge visualization: {max_file}")
            
        except Exception as e:
            print(f"Error creating mean/max visualizations: {e}")

    def _get_max_surge_data(self):
        """Get maximum surge heights across all ensemble members."""
        try:
            # Load individual ensemble member files
            surge_dir = Path("data/preprocessed/surge")
            max_surge = None
            
            # Find all individual member files
            member_files = list(surge_dir.glob("nicaragua_surge_ensemble_member_*.tif"))
            
            if not member_files:
                print("No individual member files found, using ensemble mean for max")
                return None
            
            print(f"Loading {len(member_files)} ensemble members for max calculation...")
            
            # Load each member and track the maximum
            for i, member_file in enumerate(member_files):
                try:
                    import rasterio
                    with rasterio.open(member_file) as src:
                        member_data = src.read(1)  # Read first band
                        member_data = np.flipud(member_data)  # Flip for correct orientation
                        
                        if max_surge is None:
                            max_surge = member_data.copy()
                        else:
                            max_surge = np.maximum(max_surge, member_data)
                            
                except Exception as e:
                    print(f"Warning: Could not load {member_file}: {e}")
                    continue
            
            if max_surge is None:
                return None
            
            # Apply the same inland penetration logic as the mean
            max_surge_interpolated = self._apply_inland_penetration_to_data(max_surge)
            
            return max_surge_interpolated
            
        except Exception as e:
            print(f"Error getting max surge data: {e}")
            return None

    def _apply_inland_penetration_to_data(self, surge_data):
        """Apply inland penetration logic to surge data."""
        # This method applies the same inland penetration logic as _get_interpolated_surge_data
        # but to any surge data array (mean, max, individual members, etc.)
        
        # Create computational grid coordinates
        comp_minx, comp_maxx = -89.93, -73.83
        comp_miny, comp_maxy = 7.78, 18.97
        
        comp_lons = np.linspace(comp_minx, comp_maxx, 349)
        comp_lats = np.linspace(comp_miny, comp_maxy, 249)
        
        comp_lon_mesh, comp_lat_mesh = np.meshgrid(comp_lons, comp_lats)
        
        # Get Nicaragua bounds for target grid
        nicaragua_gdf = get_nicaragua_boundary()
        minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
        
        resolution = 0.1
        lons = np.arange(minx, maxx + resolution, resolution)
        lats = np.arange(miny, maxy + resolution, resolution)
        
        # Create target grid points
        grid_lons = []
        grid_lats = []
        for j in range(len(lats) - 1):  # latitude (rows)
            for i in range(len(lons) - 1):  # longitude (columns)
                grid_lons.append((lons[i] + lons[i+1]) / 2)
                grid_lats.append((lats[j] + lats[j+1]) / 2)
        
        # Find significant surge points (threshold approach)
        surge_threshold = get_config_value(self.config, "surge.surge_threshold_m", 0.5)
        valid_surge_mask = surge_data > surge_threshold
        comp_lons_surge = comp_lon_mesh[valid_surge_mask]
        comp_lats_surge = comp_lat_mesh[valid_surge_mask]
        comp_surge_surge = surge_data[valid_surge_mask]
        
        # Simple surge extension approach
        max_extension_distance_km = get_config_value(self.config, "surge.max_inland_distance_km", 10.0)
        extended_surge = np.zeros(len(grid_lons))
        
        # For each target grid point, find the nearest surge point and extend if within range
        for i, (lon, lat) in enumerate(zip(grid_lons, grid_lats)):
            # Find the nearest surge point
            if len(comp_lons_surge) > 0:
                distances = np.sqrt((lon - comp_lons_surge)**2 + (lat - comp_lats_surge)**2) * 111.0  # Convert to km
                nearest_idx = np.argmin(distances)
                nearest_distance_km = distances[nearest_idx]
                nearest_surge = comp_surge_surge[nearest_idx]
                
                # Only extend if within the extension distance
                if nearest_distance_km <= max_extension_distance_km:
                    # Apply distance-based decay
                    decay_factor = 1.0 - (nearest_distance_km / max_extension_distance_km)
                    extended_surge[i] = nearest_surge * decay_factor
            else:
                extended_surge[i] = 0.0
        
        # Reshape to grid
        reshaped_surge = extended_surge.reshape(len(lats)-1, len(lons)-1)
        
        return reshaped_surge

    def _get_cache_path(self) -> Path:
        """Get the cache file path for surge results."""
        if self.cache_dir is None:
            cache_dir = get_config_value(self.config, "impact_analysis.output.cache_directory", "data/results/impact_analysis/cache/")
        else:
            cache_dir = self.cache_dir
            
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Create a unique cache key based on hurricane file and parameters
        import hashlib
        cache_key = f"surge_ensemble_{hashlib.md5(str(self.hurricane_file).encode()).hexdigest()[:8]}"
        
        return cache_path / f"{cache_key}.pkl"
        
    def _save_cached_results(self):
        """Save surge ensemble results to cache."""
        if not self.results:
            return
            
        cache_path = self._get_cache_path()
        
        import pickle
        cache_data = {
            'results': self.results,
            'grid_lats': self.grid_lats,
            'grid_lons': self.grid_lons,
            'grid_depths': self.grid_depths,
            'hurricane_file': self.hurricane_file
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
            
        print(f"Surge results cached to: {cache_path}")
        
    def _load_cached_results(self) -> bool:
        """Load surge ensemble results from cache."""
        if not self.use_cache:
            return False
            
        cache_path = self._get_cache_path()
        
        if not cache_path.exists():
            return False
            
        try:
            import pickle
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Verify the cache is for the same hurricane file
            if cache_data.get('hurricane_file') != self.hurricane_file:
                print("Cache mismatch - hurricane file changed")
                return False
                
            self.results = cache_data['results']
            self.grid_lats = cache_data['grid_lats']
            self.grid_lons = cache_data['grid_lons']
            self.grid_depths = cache_data['grid_depths']
            
            return True
            
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False
