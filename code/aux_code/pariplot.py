# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:02:23 2023

@author: vwgei
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from geopy.distance import geodesic
from sklearn.cluster import KMeans
import warnings
from mpl_toolkits.basemap import Basemap

warnings.filterwarnings("ignore")

import seaborn as sns

# Step 2: Create a function to find the nearest original cluster for a new latitude/longitude pair
def assign_to_cluster(new_lat, new_lon, centroids):
    # Calculate the geodesic distance between the new point and each centroid
    distances = []
    for centroid in centroids:
        # Calculate geodesic distance (more accurate for lat/lon data)
        dist = geodesic((new_lat, new_lon), (centroid[0], centroid[1])).kilometers
        distances.append(dist)
    
    # Get the index of the closest original cluster
    closest_cluster = np.argmin(distances)

    return closest_cluster

def calculate_q_statistic(data, y_series, cluster_column):
    # Calculate overall mean
    overall_mean = y_series.mean()
    
    # Group by strata (spatial_cluster)
    grouped = data.groupby(cluster_column)

    # Initialize SSW and SST
    SSW = 0
    N = len(data)  # Total number of observations
    SST = np.sum((y_series - overall_mean) ** 2)
    
    # Calculate SSW
    for cluster_id, group in grouped:
        N_h = len(group)  # Number of units in stratum h
        stratum_mean = y_series[group.index].mean()  # Mean of the current stratum
        stratum_variance = y_series[group.index].var(ddof=0)  # Population variance
        
        SSW += N_h * stratum_variance

    # Calculate Q-statistic
    q_stat = 1 - (SSW / SST)
    
    return q_stat, SSW, SST

def calculate_and_inverse_transform_kde(data, transform=True, min_val=1, max_val=10):
    """
    Calculate the KDE of the input data array and apply an inverse transformation.
    
    Parameters:
        data (np.ndarray): The input array containing the values.
        transform (bool): Perform a log transform on the data.
        min_val (int): The minimum sample weight value
        max_val (int): The maximum sample weight value

        
    Returns:
        original_data (np.ndarray): The original input data.
        inverse_kde_values (np.ndarray): The inverse transformed KDE values for each original data point.
        scaled_kde_vales (np.ndarray): The original KDE scaled between min_val and max_val
    """
    # Drop NA values and ensure data is a 1D array
    data = data[np.isfinite(data)]

    if transform:
        # data = np.log1p(np.log1p(data)) # too extreme...
        data = np.log1p(data)

    
    # Step 1: Calculate KDE using scipy
    kde = gaussian_kde(data)
    
    # Step 2: Evaluate the KDE at the original data points
    kde_values = kde(data)
    
    # Step 3: Scale the KDE values from 1 to 10
    min_kde = np.min(kde_values)
    max_kde = np.max(kde_values)

    scaled_kde_values = min_val + (kde_values - min_kde) * (max_val - min_val) / (max_kde - min_kde)
    
    # Step 4: Apply inverse transformation
    inverse_kde_values = max_val - (scaled_kde_values - min_val) * (max_val - min_val) / (max_val - min_val)  # Simplified to: 10 - (scaled_kde_values - 1)

    return data, inverse_kde_values, scaled_kde_values

# def calculate_morans_I(df, residuals, threshold=1):
#     # Pulling the LAT and LON values corresponding to the indices in x_test
#     coordinates = list(zip(df['Latitude_0'], df['Longitude_0']))

#     # Create a spatial weights matrix using DistanceBand (can be adjusted to other weight schemes)
#     w = DistanceBand(coordinates, threshold=threshold, binary=True, p=2)

#     # Normalize the spatial weights (optional, but common)
#     w.transform = 'R'

#     # Calculate Moran's I on the residuals
#     moran = Moran(residuals, w)

#     # Return the Moran's I value and p-value
#     return round(moran.I, 3), moran.p_sim

def spatial_kfold_split(data, coordinates, n_splits=5):
    # Apply KMeans clustering to create spatially-based folds
    kmeans = KMeans(n_clusters=n_splits, random_state=42)
    data['kfold_cluster'] = kmeans.fit_predict(coordinates)
    return data

def calc_stats(data):
    # Calculate lat/lon interaction and differences
    data = data.assign(
        lat_lon_interaction_0=data['Longitude_0'] * data['Latitude_0'],
        lat_diff=data['Latitude_0'] - data['Latitude_-24'],
        lon_diff=data['Longitude_0'] - data['Longitude_-24'],
        straight_line_efficiency_ratio=data['Dist_from_origin_-24'] / data['Cumulative_Dist_-24']
    )

    # Dew Point and LCL Height calculation
    relative_humidity_values = data[[f'Relative_Humidity_{i}' for i in range(-24, 1)]].values
    temperature_values = data[[f'Temperature_C_{i}' for i in range(-24, 1)]].values
    dew_point_values = temperature_values - ((100 - relative_humidity_values) / 5)
    for i, timestep in enumerate(range(-24, 1)):
        data[f'dew_point_{timestep}'] = dew_point_values[:, i]

    # Calculate LCL from dew point and temperature
    lcl_values = (temperature_values - dew_point_values) / 8 * 1000
    for i, timestep in enumerate(range(-24, 1)):
        data[f'LCL_{timestep}'] = lcl_values[:, i]

    moisture_flux_values = data[[f'Moisture_Flux_{i}' for i in range(-24, 1)]].values

    # Rainfall Estimation based on Relative Humidity and Moisture Flux
    rh_threshold = 80
    rainfall_estimation = moisture_flux_values * (relative_humidity_values / 100)
    rainfall_estimation[relative_humidity_values <= rh_threshold] = 0
    for i, timestep in enumerate(range(-24, 1)):
        data[f'Rainfall_Estimate_{timestep}'] = rainfall_estimation[:, i]

    # Add extra features to main data
    # Step 1: Convert DateTime_0 to datetime format
    for i in range(-24, 1):
        data[f'DateTime_{i}'] = pd.to_datetime(data[f'DateTime_{i}'], format="%m/%d/%Y %H:%M")

        # Step 2: Define a fixed start time (January 1, 1900)
        start_time = pd.Timestamp("1900-01-01 00:00:00")
        # Step 3: Calculate the time difference from this fixed start time
        data[f'time_difference_{i}'] = data[f'DateTime_{i}'] - start_time  # Calculate the difference

        # Step 4: Convert the difference to total seconds
        data[f'time_difference_seconds_{i}'] = data[f'time_difference_{i}'].dt.total_seconds()

        # Step 2: Extract the hour and encode it as cyclical features
        data[f'hour_sin_{i}'] = np.sin(2 * np.pi * data[f'DateTime_{i}'].dt.hour / 24)
        data[f'hour_cos_{i}'] = np.cos(2 * np.pi * data[f'DateTime_{i}'].dt.hour / 24)

        # data['month'] = data['DateTime_0'].dt.month
        # Step 2: Extract the hour and encode it as cyclical features
        data[f'month_sin_{i}'] = np.sin(2 * np.pi * data[f'DateTime_{i}'].dt.month / 12)
        data[f'month_cos_{i}'] = np.cos(2 * np.pi * data[f'DateTime_{i}'].dt.month / 12)

        # Summary Statistics for each field
    fields_to_summarize = [
        'Pressure', 'Potential_Temperature', 'Relative_Humidity', 'Specific_Humidity',
        'Solar_Radiation', 'Mixing_Depth', 'Moisture_Flux', 'Temperature_C', 'Rainfall_Estimate',
        'LCL', 'dew_point', 'Latitude', 'Longitude'
    ]
    for field in fields_to_summarize:
        print(f'Calculating stats for {field}...')
        cols = [f'{field}_{i}' for i in range(-24, 0)]
        data[f'std_{field.lower()}'] = data[cols].std(axis=1)
        data[f'delta_{field.lower()}'] = data[f'{field}_-1'] - data[f'{field}_-24']
        data[f'mean_{field.lower()}'] = data[cols].mean(axis=1)
        data[f'sum_{field.lower()}'] = data[cols].sum(axis=1)

    return data

# Define a function to calculate the radius of curvature
def calc_radius_of_curvature(bearing_current, bearing_next, ptp_dist):
    """
    Calculate the radius of curvature given the current and next bearings (in radians) and the cumulative distance (in meters).
    Bearings should be in radians, and cumulative_dist in meters.
    """ 
    # # Convert cumulative distance from meters to kilometers
    # cumulative_dist = cumulative_dist / 1000  # Meters to kilometers

    # Calculate the change in bearing, ensuring the result is between -pi and pi
    delta_theta = (bearing_next - bearing_current + np.pi) % (2 * np.pi) - np.pi

    # Prevent division by zero by checking if delta_theta is 0
    if delta_theta == 0:
        return np.inf  # Infinite radius, straight line

    # Radius of curvature (arc length / change in angle)
    radius = ptp_dist / delta_theta

    return radius


# Define a function to calculate angular velocity
def calc_angular_velocity(bearing_current, bearing_next, delta_t=1):
    """
    Calculate the angular velocity given the current and next bearings and the time interval.
    Bearings should be in radians and delta_t in hours.
    """
    # Calculate the change in bearing, ensuring the result is between -pi and pi
    delta_theta = (bearing_next - bearing_current + np.pi) % (2 * np.pi) - np.pi
    
    # Angular velocity (radians per hour)
    w = delta_theta / delta_t
    
    return w

R_d = 287.05  # Specific gas constant for dry air in J/(kg·K)
def saturation_vapor_pressure(temperature):
    """
    Calculate saturation vapor pressure using the Tetens formula.
    Temperature in Celsius.
    """
    return 6.112 * np.exp((17.67 * temperature) / (temperature + 243.5))

def clausis_clapeyron(temperature):
    """
    Calculate saturation vapor pressure using Clausis-Clapeyron equation.
    Temperature in Celcius

    example call: clausis_clapeyron(293.15)
    """
    temperature += 273.15

    return 611 * np.exp((((2.5 * 10**6)/461.5) * ((1/273.15) - (1/temperature))))

# Function to calculate moist air density
def moist_air_density(temperature, relative_humidity, pressure):
    """
    Calculate the density of moist air.
    
    Parameters:
    - temperature: Temperature in Celsius
    - relative_humidity: Relative humidity as a percentage (0-100)
    - pressure: Total atmospheric pressure in hPa
    
    Returns:
    - The density of moist air in kg/m^3.
    """
    
    # Convert Pressure to hPa
    pressure = pressure * 100
    
    # Calculate saturation vapor pressure using Clausius-Clapeyron
    e_s = clausis_clapeyron(temperature)
    
    # Calculate actual vapor pressure using relative humidity
    p_v = (relative_humidity / 100.0) * e_s
    
    # Calculate the partial pressure of dry air
    p_d = pressure - p_v
    
    # Constants
    R_d = 287.05  # Specific gas constant for dry air (J/kg·K)
    R_v = 461.5   # Specific gas constant for water vapor (J/kg·K)

    # Convert temperature from Celsius to Kelvin
    temp_k = temperature + 273.15
    
    # Calculate the density of moist air
    density = (p_d / (R_d * temp_k)) + (p_v / (R_v * temp_k))
    
    return density


def mass_of_moist_particle(T, P, RH):
    """
    Calculate the mass of a moist atmospheric particle given temperature, pressure, and humidity.
    
    Parameters:
    T : float
        Temperature in degrees Celsius.
    P : float
        Total pressure in hPa.
    RH : float
        Relative humidity as a fraction (0 to 1).
        
    Returns:
    float
        Mass of a moist particle in kg.
    """
    # Calculate saturation vapor pressure
    P_sat = clausis_clapeyron(T)  # in hPa

    RH = RH / 100
    
    # Calculate partial pressures
    P_water = RH * P_sat  # Partial pressure of water vapor in hPa
    P_dry = P - P_water   # Partial pressure of dry air in hPa
    
    # Molar masses (in g/mol)
    M_d = 28.94  # Molar mass of dry air
    M_w = 18.02  # Molar mass of water vapor
    
    # Calculate average molar mass of moist air (in g/mol)
    M_avg = (M_d * P_dry + M_w * P_water) / P  # in g/mol
    
    # Convert to kg/mol
    M_avg_kg = M_avg / 1000  # Convert g/mol to kg/mol
    
    # Calculate mass of a single particle (in kg)
    N_A = 6.022e23  # Avogadro's number
    m_particle = M_avg_kg / N_A  # in kg
    
    return m_particle


# Define a function to calculate moment of inertia
def calc_moment_of_inertia(radius_of_curvature, temperature, pressure, relative_humidity):
    # mass of an average air particle
    # assign a theoretical mass to the particle
    # mass = 4.81 * 10**-26 # (kg)

    # Moment of inertia
    I = mass_of_moist_particle(temperature, pressure, relative_humidity) * radius_of_curvature**2
    
    return I

# Define a function to calculate angular momentum
def calc_angular_momentum(moment_of_inertia, angular_velocity):
    """
    Calculate the angular momentum given the moment of inertia and angular velocity.
    """
    L = moment_of_inertia * angular_velocity

    return L

def calc_velocity(distance, delta_t=1):
    """Calculate velocity in m/s."""
    return (distance / 1000) / (delta_t / 3600)  # Convert distance from meters to kilometers and delta_t from hours to seconds

def calc_acceleration(velocity_0, velocity_1, delta_t=1):
    return (velocity_1 - velocity_0) / delta_t

omega_earth = 7.2921e-5  # Earth's angular velocity in rad/s
def calc_coriolis_acceleration(lat, velocity):
    """
    Calculate Coriolis acceleration in m/s^2 given latitude and velocity.
    Latitude is in degrees, velocity is in km/h.
    """
    # Convert latitude to radians
    lat_rad = np.deg2rad(lat)

    
    # Calculate Coriolis acceleration
    coriolis_acc = 2 * omega_earth * np.sin(lat_rad) * velocity
    
    return coriolis_acc

def haversine_calc(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    # Radius of Earth in meters
    R = 6371000

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance  # Distance in meters

def haversine(lat_lon1, lat_lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = np.radians(lat_lon1)
    lat2, lon2 = np.radians(lat_lon2)

    # Radius of Earth in meters
    R = 6371000

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance

def calculate_wind_speed(lat1, lon1, lat2, lon2, duration_hours=1):
    # Calculate the distance between points in meters
    distance = haversine(lat1, lon1, lat2, lon2)
    
    # Convert hours to seconds
    duration_seconds = duration_hours * 3600
    
    # Calculate wind speed in m/s
    wind_speed = distance / duration_seconds
    
    return wind_speed

def calc_rossby_number(acceleration, wind_speed, lat):
    """
    Calculate the Rossby number given velocity, radius of curvature, and latitude.
    """

    # Coriolis parameter f
    f = 2 * omega_earth * np.sin(np.deg2rad(lat))
    
    if f == 0 or wind_speed == 0:
        return np.inf  # Infinite Rossby number for zero Coriolis effect or zero radius
    
    Ro = acceleration / (f * wind_speed)
    return Ro


g = 9.81  # gravitational acceleration in m/s^2
def calc_brunt_vaisala(pot_temp1, pot_temp2, height1, height2):
    """
    Calculate the Brunt-Väisälä frequency given two points with potential temperatures and heights.
    """
    # Calculate the change in potential temperature and height
    delta_theta = pot_temp2 - pot_temp1
    delta_z = height2 - height1  # Height difference in meters
    
    # Avoid division by zero
    if delta_z == 0 or pot_temp1 == 0:
        return 0
    
    # Calculate N^2 (in s^-2)
    N_squared = (g / pot_temp1) * (delta_theta / delta_z)
    
    # Return N (Brunt-Väisälä frequency)
    return np.sqrt(N_squared)


def calc_divergence(lat1, lon1, lat2, lon2, v1, v2):
    """
    Calculate divergence between two points using the velocity change and distance between them.
    
    Parameters:
    lat1, lon1 : float
        Latitude and longitude of the first point.
    lat2, lon2 : float
        Latitude and longitude of the second point.
    v1, v2 : float
        Velocity at the first and second points in meters per second (m/s).
        
    Returns:
    float
        Divergence in s^{-1}. Returns NaN if any input is NaN or if the distance is zero.
    """
    # Check for NaN values
    if np.isnan(lat1) or np.isnan(lon1) or np.isnan(lat2) or np.isnan(lon2) or np.isnan(v1) or np.isnan(v2):
        return np.nan  # or return 0 if you prefer a specific value
    
    # Calculate distance between two points (in meters)
    dist = geodesic((lat1, lon1), (lat2, lon2)).meters  # Change to meters
    
    # Avoid division by zero
    if dist == 0:
        return 0
    
    # Divergence = (velocity change) / (distance between points)
    divergence = (v2 - v1) / dist  # This is in s^{-1}

    return divergence

def dynamic_viscosity_air(temperature_c, relative_humidity):
    # Convert temperature to Kelvin
    temperature_k = temperature_c + 273.15

    # Sutherland's constant for air
    mu_dry = 1.78e-5  # Dynamic viscosity of dry air at 0°C in Pa·s
    T0 = 273.15  # Reference temperature in K
    S = 111  # Sutherland's constant for air

    # Calculate dynamic viscosity of dry air using Sutherland's formula
    mu_dry_temp = mu_dry * ((T0 + S) / (temperature_k + S)) * (temperature_k / T0) ** (3/2)

    # Empirical relationship for effect of humidity (approx)
    k = 0.02  # Example constant, adjust based on specific conditions
    mu_moist = mu_dry_temp * (1 + k * relative_humidity)

    return mu_moist

def calculate_reynolds_number(pressure, temperature_c, relative_humidity, distance=1):
    """Calculate Reynolds number using input parameters."""
    temperature_k = temperature_c + 273.15  # Convert °C to K

    # Calculate density of moist air kg/m^3
    density = moist_air_density(pressure, temperature_k, relative_humidity)

    # Calculate dynamic viscosity of moist air
    viscosity = dynamic_viscosity_air(temperature_c, relative_humidity)

    # Calculate velocity (ensure this is in m/s)
    velocity = calc_velocity(distance)  # Ensure distance is in meters

    # Calculate Reynolds number
    reynolds_number = (density * velocity * distance) / viscosity
    return reynolds_number

def calc_physics_fields(data: pd.DataFrame) -> pd.DataFrame:
    # # Calculate Radius of Curvature
    # for i in range(-24, 0):
    #     bearings_current_col = f'bearings_ptp_{i}'
    #     bearings_next_col = f'bearings_ptp_{i + 1}'
    #     cumulative_dist_col = f'Distance_ptp_{i}'
    #     data[f'Radius_of_Curvature_{i}'] = data.apply(
    #         lambda row: calc_radius_of_curvature(row[bearings_current_col], row[bearings_next_col], row[cumulative_dist_col]), axis=1)

    # # Calculate Angular Velocity
    # for i in range(-24, 0):
    #     bearings_current_col = f'bearings_ptp_{i}'
    #     bearings_next_col = f'bearings_ptp_{i + 1}'
    #     data[f'Angular_Velocity_{i}'] = data.apply(
    #         lambda row: calc_angular_velocity(row[bearings_current_col], row[bearings_next_col]), axis=1)

    # # Calculate Moment of Inertia
    # for i in range(-24, 0):
    #     pressure_col = f'Pressure_{i}'
    #     temperature_col = f'Temperature_C_{i}'
    #     rh_col = f'Relative_Humidity_{i}'
    #     radius_col = f'Radius_of_Curvature_{i}'
    #     data[f'Moment_of_Inertia_{i}'] = data.apply(
    #         lambda row: calc_moment_of_inertia(row[radius_col], row[pressure_col], row[temperature_col], row[rh_col]), axis=1)

    # # Calculate Angular Momentum
    # for i in range(-24, 0):
    #     angular_velocity_current_col = f'Angular_Velocity_{i}'
    #     moment_of_inertia_current_col = f'Moment_of_Inertia_{i}'
    #     data[f'Angular_Momentum_{i}'] = data.apply(
    #         lambda row: calc_angular_momentum(row[moment_of_inertia_current_col], row[angular_velocity_current_col]), axis=1)

    # Calculate Velocity
    print("Calculating Velocity")
    for i in range(-24, 0):
        distance_col = f'Distance_ptp_{i}'
        data[f'Velocity_{i}'] = data[distance_col].apply(calc_velocity)
    
    # print("Calculating Acceleration...")
    # # Loop over the time steps and calculate velocity
    # for i in range(-24, -1):  # From -24 to -1
    #     velocity_col = f'Velocity_{i}'
    #     next_velocity_col = f'Velocity_{i+1}'
    #     data[f'Acceleration_{i}'] = data.apply(lambda row: calc_acceleration(row[velocity_col],
    #                                                                 row[next_velocity_col]), axis=1)
    #     # print(data[f'Velocity_{i}'])
    # print("Done!")


    # print("Calculating Coriolis Acceleration...")
    # # Loop through time steps and calculate Coriolis acceleration
    # for i in range(-24, 0):
    #     lat_col = f'Latitude_{i}'  # Assuming a column that contains the latitude
    #     velocity_col = f'Velocity_{i}'
        
    #     data[f'Coriolis_Accel_{i}'] = data.apply(lambda row: calc_coriolis_acceleration(row[lat_col], 
    #                                                                                 row[velocity_col]), axis=1)
    # print("Done!")

    # # Example loop to calculate wind speed at each time step
    # print("Calculating Wind Speed...")
    # for i in range(-24, -1):
    #     # Define column names for latitude and longitude at time i and i+1
    #     lat_col_1 = f'Latitude_{i}'
    #     lon_col_1 = f'Longitude_{i}'
    #     lat_col_2 = f'Latitude_{i+1}'
    #     lon_col_2 = f'Longitude_{i+1}'
        
    #     # Calculate distance and then speed
    #     data[f'Wind_Speed_{i}'] = data.apply(
    #         lambda row: haversine(row[lat_col_1], row[lon_col_1], row[lat_col_2], row[lon_col_2]) / 3600
    #         if pd.notnull(row[lat_col_1]) and pd.notnull(row[lon_col_1]) and
    #         pd.notnull(row[lat_col_2]) and pd.notnull(row[lon_col_2])
    #         else np.nan,
    #         axis=1
    #     )

    # print("Done!")
                        
    # print("Calculating Rossby Number...")
    # # Loop through the time steps to calculate Rossby Number
    # for i in range(-24, 0):
    #     lat_col = f'Latitude_{i}'  # Assuming a column for latitude
    #     acceleration_col = f'Acceleration_{i}'
    #     windspeed_col = f'Wind_Speed_{i}'  # From earlier
        
    #     data[f'Rossby_Number_{i}'] = data.apply(lambda row: calc_rossby_number(row[velocity_col],
    #                                                                         row[radius_col],
    #                                                                         row[lat_col]), axis=1)
    # print("Done!")

    # # Calculate Brunt-Väisälä Frequency
    # print("Calculating BV")
    # for i in range(-24, 0):
    #     pot_temp1_col = f'Potential_Temperature_{i}'
    #     pot_temp2_col = f'Potential_Temperature_{i + 1}'
    #     height1_col = f'AltP_meters_{i}'
    #     height2_col = f'AltP_meters_{i + 1}'
    #     data[f'Brunt_Vaisala_Freq_{i}'] = data.apply(
    #         lambda row: calc_brunt_vaisala(row[pot_temp1_col], row[pot_temp2_col], row[height1_col], row[height2_col]), axis=1)

    # print("Calculating div")
    # # Calculate Divergence
    # for i in range(-24, -1):
    #     lat1_col = f'Latitude_{i}'
    #     lon1_col = f'Longitude_{i}'
    #     lat2_col = f'Latitude_{i + 1}'
    #     lon2_col = f'Longitude_{i + 1}'
    #     v1_col = f'Velocity_{i}'
    #     v2_col = f'Velocity_{i + 1}'
    #     data[f'Divergence_{i}'] = data.apply(
    #         lambda row: calc_divergence(row[lat1_col], row[lon1_col], row[lat2_col], row[lon2_col], row[v1_col], row[v2_col]), axis=1)

    # # Calculate Moist Air Density
    # for i in range(-24, 0):
    #     pressure_col = f'Pressure_{i}'
    #     temperature_col = f'Temperature_C_{i}'
    #     rh_col = f'Relative_Humidity_{i}'
    #     data[f'Density_Moist_{i}'] = data.apply(
    #         lambda row: moist_air_density(row[temperature_col], row[rh_col], row[pressure_col]), axis=1)

    # # Calculate Reynolds Number
    # for i in range(-24, 0):
    #     pressure_col = f'Pressure_{i}'
    #     temperature_col = f'Temperature_C_{i}'
    #     rh_col = f'Relative_Humidity_{i}'
    #     data[f'Reynolds_Number_{i}'] = data.apply(
    #         lambda row: calculate_reynolds_number(row[pressure_col], row[temperature_col], row[rh_col]), axis=1)

    # Summary Statistics for each field
    fields_to_summarize = [
        'Velocity', #'Divergence'
        # 'Radius_of_Curvature', 'Angular_Velocity', 'Angular_Momentum', 'Velocity', 'Coriolis_Accel',
        # 'Rossby_Number', 'Brunt_Vaisala_Freq', 'Divergence', 'Density_Moist', 'Reynolds_Number'
    ]
    for field in fields_to_summarize:
        cols = [f'{field}_{i}' for i in range(-24, 0)]
        if field == "Divergence":
            cols = [f'{field}_{i}' for i in range(-24, -1)]
        data[f'std_{field.lower()}'] = data[cols].std(axis=1)
        if field == "Divergence":
            data[f'delta_{field.lower()}'] = data[f'{field}_-2'] - data[f'{field}_-24']
        else:
            data[f'delta_{field.lower()}'] = data[f'{field}_-1'] - data[f'{field}_-24']
        data[f'mean_{field.lower()}'] = data[cols].mean(axis=1)

    return data


# Function to perform KMeans clustering on a specific label subset and assign new cluster labels
def perform_kmeans_for_label(df, label, kmeans_k, rs, label_threshold):
    # Filter the dataframe for the current label
    subset = df[df['spatial_cluster'] == label]
    
    # Extract the coordinates for the subset
    coordinates_subset = subset[['Latitude_0', 'Longitude_0']]
    
    # Adjust longitude if needed
    coordinates_subset.loc[coordinates_subset['Longitude_0'] > 100, 'Longitude_0'] -= 360
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=kmeans_k, random_state=rs)
    new_labels = kmeans.fit_predict(coordinates_subset)
    
    # # Assign a unique label to each cluster within this subset, starting from the threshold
    new_labels += label_threshold  # This shifts the labels to start at the threshold value

    # print(np.unique(new_labels))
    
    # Directly update the 'spatial_cluster' column in the original dataframe
    df.loc[subset.index, 'spatial_cluster'] = new_labels
    
    return df

# Read in CSV file with all of the data we need. Meteorology variables + Pathdata + VOC data
file_path = r"C:\Users\vwgei\Documents\PVOCAL\data\V4\V4_Awakens.csv" #"C:\Users\vwgei\Documents\PVOCAL\data\A_NEW_ERA5_PASTEL.csv"

# Load data from CSV into a pandas DataFrame.
df = pd.read_csv(file_path)

# file_path_oc = r"C:\Users\vwgei\Documents\PVOCAL\data\V4\GDAS_A_NEW_HOPE.csv"

# data_oc = pd.read_csv(file_path_oc)

# Replace WAS nodata value with np.nan for consistency
df.replace(-999999.0, np.nan, inplace=True)
df.replace(-888888.0, np.nan, inplace=True)
df.replace(-888888888.0, np.nan, inplace=True)
df.replace(-999999999.0, np.nan, inplace=True)
df.replace(-99999.000000, np.nan, inplace=True)
df.replace(-999.990000, np.nan, inplace=True)
df.replace(-9.999999e+09, np.nan, inplace=True)
df.replace(-9.999990e+08, np.nan, inplace=True)
df.replace(-8.888888e+06, np.nan, inplace=True)
df.replace(-9.999999e+06, np.nan, inplace=True)

# Replace WAS nodata value with np.nan for consistency
df.replace(-8888, np.nan, inplace=True)
df.replace(-999, np.nan, inplace=True)
df.replace(-888, np.nan, inplace=True)
df.replace(-777, np.nan, inplace=True)
df.replace(-777.770000, np.nan, inplace=True)
df.replace(-888.800000, np.nan, inplace=True)
df.replace(-888.880000, np.nan, inplace=True)
df.replace(10000000, np.nan, inplace=True)

# Replace values in df['DMS_WAS'] lower than 1 with np.nan
df.loc[df['DMS'] < 1, 'DMS'] = np.nan
df.loc[df['O3'] < 1, 'O3'] = np.nan
df.loc[df['CH4'] < 1500, 'CH4'] = np.nan

df = calc_stats(df)
df = calc_physics_fields(df)

# datat0 = data.iloc[:,2:13]

# datat24 = data.iloc[:,13:26]

# dataVOC = data.iloc[:,34:134]
# 201 #211
# 213 #217

input_features = [
    'mean_distance_to_cluster_centroids', 'mean_distance_to_anitmeridian', 'mean_distance_to_t1000_cities',
    'min_distance_to_t1000_cities', 'min_distance_to_cluster_centroids', 'min_distance_to_extent_corners',
    'Specific_Humidity_0', 'Potential_Temperature_0', 'mean_lcl', 'bearings_from_origin_-24',
    'sum_moisture_flux', 'mean_mixing_depth', 'sum_solar_radiation', 'domain_indicator', 'time_difference_seconds_0',
    'DMS', 'Ethane', 'CO', 'CH4', 'O3', 'CH3Br',
]

# Define constants
kmeans_dist = 100
rs = 42  # Set a random seed for reproducibility if needed

# Convert lat/lon to radians for both points and centroids
coordinates = df[['Latitude_0', 'Longitude_0']].copy()
coordinates.loc[coordinates['Longitude_0'] > 100, 'Longitude_0'] -= 360
coords_radians = np.radians(coordinates.values)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=kmeans_dist, random_state=rs)
kmeans.fit(coordinates)
centroids = kmeans.cluster_centers_
centroids_radians = np.radians(centroids)  # Convert centroids to radians

# Haversine function
def haversine(lat_lon1, lat_lon2):
    R = 6371  # Radius of Earth in meters
    dlat = lat_lon2[0] - lat_lon1[0]
    dlon = lat_lon2[1] - lat_lon1[1]
    a = np.sin(dlat / 2)**2 + np.cos(lat_lon1[0]) * np.cos(lat_lon2[0]) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Parallelized function to calculate distances from one point to all centroids
def distance_to_centroids(point, centroids):
    return [haversine(point, centroid) for centroid in centroids]

# Run in parallel: calculate distance from each point to all centroids
distances = Parallel(n_jobs=10)(delayed(distance_to_centroids)(coords_radians[i], centroids_radians) for i in range(len(coords_radians)))
dist_matrix = np.array(distances)

# Optional: Compute minimum and mean distance to centroids as features
df['min_distance_to_cluster_centroids'] = dist_matrix.min(axis=1)
df['mean_distance_to_cluster_centroids'] = dist_matrix.mean(axis=1)

world_cites = pd.read_csv(r"C:\Users\vwgei\Documents\PVOCAL\data\worldcities.csv")
top_1000 = world_cites.sort_values('population', ascending=False).iloc[:1000]

top_1000_ll = top_1000[['lat', 'lng']]
top_1000_ll.loc[top_1000_ll['lng'] > 100, 'lng'] -= 360
top_1000_radians = np.radians(top_1000_ll.values)

# Run in parallel: calculate distance from each point to all centroids
distances_cities = Parallel(n_jobs=10)(delayed(distance_to_centroids)(coords_radians[i], top_1000_radians) for i in range(len(coords_radians)))
dist_matrix_cities = np.array(distances_cities)

# Optional: Compute minimum and mean distance to centroids as features
df['min_distance_to_t1000_cities'] = dist_matrix_cities.min(axis=1)
df['mean_distance_to_t1000_cities'] = dist_matrix_cities.mean(axis=1)

point_90_lonmin = (90, -247.988)

point_neg90_lonmin = (-90, -247.988)

# point_90_neg180 = (90, -143.7789)

# point_neg90_neg180 = (-90, -143.7789)

point_90_lonmax = (90, -14.42)

point_neg90_lonmax = (-90, -14.42)

# List of reference points
reference_points = [point_90_lonmin, point_neg90_lonmin, point_90_lonmax, point_neg90_lonmax]

# Step 2: Define Function to Calculate Distances to Each Reference Point
def calculate_distances_to_references(point):
    """Calculate the distances from a given point to each reference point."""
    distances = [geodesic(point, ref_point).kilometers for ref_point in reference_points]
    return distances

# Step 3: Run Distance Calculations in Parallel for Each Point in `df`
# Prepare points from df as a list of tuples (lat, lon) for each point
points = list(zip(df['Latitude_0'], df['Longitude_0']))

# Parallel computation of distance matrix
distances_extent_corners = Parallel(n_jobs=-1)(
    delayed(calculate_distances_to_references)(point) for point in points
)
dist_matrix_extents = np.array(distances_extent_corners)

# Step 4: Compute Minimum and Mean Distances for Each Point and Add to `df`
df['min_distance_to_extent_corners'] = dist_matrix_extents.min(axis=1)
df['mean_distance_to_extent_corners'] = dist_matrix_extents.mean(axis=1)

# Step 1: Define Reference Points
# A) Average dataset center (mean latitude and longitude)
# point_90_lonmin = (90, -247.988)

# point_neg90_lonmin = (-90, -247.988)

point_90_neg180 = (90, -180)

point_0_neg180 = (0, -180)

point_neg90_neg180 = (-90, -180)

# List of reference points
reference_points = [point_90_neg180, point_0_neg180, point_neg90_neg180]

# Step 2: Define Function to Calculate Distances to Each Reference Point
def calculate_distances_to_references(point):
    """Calculate the distances from a given point to each reference point."""
    distances = [geodesic(point, ref_point).kilometers for ref_point in reference_points]
    return distances

# Step 3: Run Distance Calculations in Parallel for Each Point in `df`
# Prepare points from df as a list of tuples (lat, lon) for each point
points = list(zip(df['Latitude_0'], df['Longitude_0']))

# Parallel computation of distance matrix
distances_antimeridian = Parallel(n_jobs=-1)(
    delayed(calculate_distances_to_references)(point) for point in points
)
distances_antimeridian = np.array(distances_antimeridian)

# Step 4: Compute Minimum and Mean Distances for Each Point and Add to `df`
df['min_distance_to_antimeridian'] = distances_antimeridian.min(axis=1)
df['mean_distance_to_anitmeridian'] = distances_antimeridian.mean(axis=1)


kmeans_k = 35

# Assuming data contains 'Latitude_0' and 'Longitude_0'
coordinates = df[['Latitude_0', 'Longitude_0']]
coordinates.loc[coordinates['Longitude_0'] > 100, 'Longitude_0'] -= 360

# Perform KMeans clustering
kmeans = KMeans(n_clusters=kmeans_k, random_state=rs)
df['spatial_cluster'] = kmeans.fit_predict(coordinates)


centroids = kmeans.cluster_centers_

map_center_lat = np.mean(df['Latitude_0'])
map_center_lon = np.mean(df['Longitude_0'])

angles = np.arctan2(centroids[:, 1] + 90, centroids[:, 0])
angles = np.degrees(angles) % 360

angles = (angles - 90) % 360

# Step 3: Sort clusters based on angles
sorted_indices = np.argsort(angles)
new_cluster_labels = np.zeros_like(df['spatial_cluster'])

# Step 4: Map old cluster labels to new sorted labels
for new_label, old_label in enumerate(sorted_indices):
    new_cluster_labels[df['spatial_cluster'] == old_label] = new_label

# Update the original spatial_cluster column with the sorted cluster labels
df['spatial_cluster'] = new_cluster_labels

sub_clusters = 4
# Threshold for new cluster labels
# Perform KMeans clustering for each label individually (5, 7, and 8)
labels_to_cluster = [14,15,17,18,25,26,27,28,30,32]
increment = range(kmeans_k, (kmeans_k + (sub_clusters * len(labels_to_cluster))), sub_clusters)
iterator = 0
for label in labels_to_cluster:
    df = perform_kmeans_for_label(df, label, sub_clusters, rs, increment[iterator])
    iterator += 1


# Create a mapping of unique cluster labels to sequential numbers
unique_labels = np.unique(df['spatial_cluster'])
label_mapping = {label: new_label for new_label, label in enumerate(unique_labels)}

# Apply the mapping to reassign labels
df['spatial_cluster'] = df['spatial_cluster'].replace(label_mapping)

# Step 5: Compute the centroids for the new sorted clusters
sorted_centroids = np.zeros((len(np.unique(df['spatial_cluster'])) , 2))

# Recompute centroids based on the new sorted cluster labels
for cluster in np.unique(df['spatial_cluster']):
    cluster_data = df[df['spatial_cluster'] == cluster]
    sorted_centroids[cluster] = [cluster_data['Latitude_0'].mean(), cluster_data['Longitude_0'].mean()]


lats = df["Latitude_0"].values
longs = df["Longitude_0"].values
longs = np.where(longs > 100, longs - 360, longs)
# cluster_labels = df['spatial_cluster']
cluster_labels = df['spatial_cluster']

mapcorners = [-260, -90, 5, 90]  # Full globe

# Create a map with custom boundaries
plt.figure(figsize=(10, 8))
m = Basemap(projection='cyl', llcrnrlon=mapcorners[0], llcrnrlat=mapcorners[1],
            urcrnrlon=mapcorners[2], urcrnrlat=mapcorners[3], lat_ts=20, resolution='c', lon_0=135)

# Draw coastlines, countries, and states
m.drawcoastlines()
m.drawcountries()
m.drawstates()

# Convert latitude and longitude to map coordinates
x, y = m(longs, lats)

# Plot points
scatter = m.scatter(x, y, s=10, c=cluster_labels, cmap='gist_rainbow', edgecolor='none')

# Loop through each unique cluster label and add text at the midpoint occurrence of that label
unique_labels = np.unique(cluster_labels)
for label in unique_labels:
    # Find all the indices where the cluster label occurs
    indices = np.where(cluster_labels == label)[0]
    # Find the index of the midpoint occurrence
    midpoint_index = indices[len(indices) // 2]
    # Add text label for this cluster at the midpoint location
    plt.text(x[midpoint_index], y[midpoint_index], str(label), fontsize=10, ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

# Add colorbar
plt.colorbar(scatter, fraction=0.02, pad=0.01, label="Cluster Labels")

# Draw latitude and longitude grid lines
m.drawparallels(range(-90, 91, 30), labels=[1, 0, 0, 0], fontsize=10)  # Latitude lines
m.drawmeridians(range(-180, 181, 45), labels=[0, 0, 0, 1], fontsize=10)  # Longitude lines

plt.title('Spatial Clusters')
plt.savefig(r"C:\Users\vwgei\Documents\PVOCAL\plots\standalone\PAIRPLOTglobal_clusters.png")
plt.show()


# Define latitude and longitude boundaries for North America
na_mapcorners = [-130, 20, -60, 60]  # North America bounding box

# Create a figure for the North America zoomed-in map
plt.figure(figsize=(10, 8))
m_na = Basemap(projection='cyl', llcrnrlon=na_mapcorners[0], llcrnrlat=na_mapcorners[1],
               urcrnrlon=na_mapcorners[2], urcrnrlat=na_mapcorners[3], resolution='c')

# Draw coastlines, countries, and states
m_na.drawcoastlines()
m_na.drawcountries()
m_na.drawstates()

# Convert latitude and longitude to map coordinates for North America plot
x_na, y_na = m_na(longs, lats)

# Plot points for North America map
scatter_na = m_na.scatter(x_na, y_na, s=10, c=cluster_labels, cmap='prism', edgecolor='none')

# Accessing the values of the keys
values = np.array([label_mapping[key] for key in label_mapping.keys()])
usa_keys = values[values > (kmeans_k - 2)]


# Add text labels at midpoint for each cluster
for label in usa_keys:
    indices = np.where(cluster_labels == label)[0]
    midpoint_index = indices[len(indices) // 2]
    plt.text(x_na[midpoint_index], y_na[midpoint_index], str(label), fontsize=10, ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

# Add colorbar for the North America map
plt.colorbar(scatter_na, fraction=0.02, pad=0.01, label="Cluster Labels")

# Draw latitude and longitude grid lines for North America map
m_na.drawparallels(range(20, 61, 10), labels=[1, 0, 0, 0], fontsize=10)  # Latitude lines
m_na.drawmeridians(range(-130, -59, 20), labels=[0, 0, 0, 1], fontsize=10)  # Longitude lines

# Title and save the zoomed-in plot
plt.title('Spatial Clusters over the USA')
plt.savefig(r"C:\Users\vwgei\Documents\PVOCAL\plots\standalone\PAIRPLOTUSA_clusters.png")
plt.show()


# df.to_csv(r"C:\Users\vwgei\Documents\PVOCAL\data\V4\df_preprocessed.csv", index=False)

from matplotlib.colors import ListedColormap

# Extract the colors from tab20b
tab20_colors = plt.cm.tab20b(np.linspace(0, 1, 20))

# Add a 21st color by interpolating (or define a custom color explicitly)
extra_color = np.array([[0.5, 0.5, 0.5, 1.0]])  # A neutral gray as an example
# extra_color = np.array([[0.0, 1.0, 1.0, 1.0]])  # Cyan: full green, full blue, no red
custom_colors = np.vstack([tab20_colors, extra_color])

# Create a custom ListedColormap
custom_tab20_21 = ListedColormap(custom_colors, name='custom_tab20_21')


# Example dictionary mapping numeric labels to custom string labels
label_mapping = {
    0: "ATom",
    1: "ACCLIP",
    2: "DC3",
    3: "DISCOVER-AQ_FRAPPE",
    4: "FIREX-AQ",
    5: "INTEX-A",
    6: "INTEX-B (C130)",
    7: "INTEX-B (DC8)",
    8: "KORUS-AQ",
    9: "PEM TROPICS A (DC8)",
    10: "PEM TROPICS A (P3)",
    11: "PEM TROPICS B (DC8)",
    12: "PEM TROPICS B (P3)",
    13: "PEM WEST A",
    14: "PEM WEST B",
    15: "SEAC4RS (DC8)",
    16: "TRACE A (DC8)",
    17: "TRACE A (P3)",
    18: "TRACE P (DC8)",
    19: "TRACE P (P3)",
    20: "WINTER",
    # Add more mappings as needed
}


global_subset = df[input_features]

# Map numeric labels in the data to human-readable labels
global_subset['domain_indicator_label'] = global_subset['domain_indicator'].map(label_mapping)

# Create a palette dictionary mapping labels to colors
custom_palette = {label_mapping[i]: custom_colors[i] for i in label_mapping.keys()}

# Create the pairplot with the custom palette and hue labels
sns.pairplot(
    global_subset,
    height=2.5,
    diag_kind='kde',
    hue='domain_indicator_label',  # Use human-readable labels
    palette=custom_palette,  # Use the custom color mapping
    plot_kws={'alpha': 0.2}  # Translucent points
)

plt.savefig(r"C:\Users\vwgei\Documents\PVOCAL\plots\PASTEL_PAIRPLOT.png", bbox_inches='tight')
plt.show()
# # # Plot Correlation coefficent using a heatmap
# # plt.figure(figsize=(20,18))
# # cor=df.corr(method='pearson')
# # cor = abs(cor)
# # sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
# # plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/CorrMatrixt0_pearson.png")
# # plt.show()



# # Define a new palette explicitly covering the full range of 'domain_indicator'
# unique_domains = range(0, 21)  # Explicitly set the range for 'domain_indicator'
# cmap = plt.cm.jet  # Colormap
# colors = cmap(np.linspace(0, 1, len(unique_domains)))  # Generate distinct colors for each value
# color_palette = dict(zip(unique_domains, colors))  # Map each domain_indicator value to a color

# Create the pairplot with the custom palette
# sns.pairplot(
#     global_subset, 
#     height=2.5, 
#     diag_kind='kde', 
#     hue='domain_indicator', 
#     palette=color_palette,  # Use the explicit color mapping
#     plot_kws={'alpha': 0.3}  # Translucent points
# )

# # Save and display the plot
# plt.savefig(r"C:\Users\vwgei\Documents\PVOCAL\plots\PASTEL_PAIRPLOT.png", bbox_inches='tight')
# plt.show()


# Plot all features and target using a pairplot
# sns.pairplot(datat24, height=2.5)
# plt.tight_layout()
# plt.show()
# plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/PairPlotT24.png")

# # Plot all features and target using a pairplot
# sns.pairplot(dataVOC)
# plt.tight_layout()
# plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/PairPlotAllVOC.png")
# plt.show()