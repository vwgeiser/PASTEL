# -*- coding: utf-8 -*-
"""
Created on Thu May  9 22:49:26 2024

@author: vwgei
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap

file_path = r"C:\Users\vwgei\Documents\PVOCAL\data\A_NEW_ERA5_PASTEL.csv"

# Load data from CSV into a pandas DataFrame.
data = pd.read_csv(file_path)

# Replace WAS nodata value with np.nan for consistency
data.replace(-8888, np.nan, inplace=True)

# # Individual VOCs
# dms = data['WAS_DMS_WAS'] 
# ethane = data['WAS_Ethane_WAS'] # GMI 92
# # isoprene = data['WAS_Isoprene_WAS']
# benzene = data['WAS_Benzene_WAS']
# # toluene = data['WAS_Toluene_WAS']
# ozone = data['UCATS-O3_O3_UCATS'] #GMI 74
# methane = data['NOAA-Picarro_CH4_NOAA'] # GMI 59
# ch3br = data['WAS_CH3Br_WAS'] # GMI 76
# h1211 = data['WAS_H1211_WAS'] # GMI 87
# h1301 = data['WAS_H1301_WAS'] # GMI 88
# CO = data['NOAA-Picarro_CO_NOAA']

data = data[(data['Mixing_Ratio_0'] == 0.0)]# | (data['Mixing_Ratio_0'] == 0.1)]

variable = data['AltP_meters_0']

# Map corners
mapcorners = [-260, -90, 20, 90]  # Full globe

ilats = data["Latitude_0"].values
ilongs = data["Longitude_0"].values

# plats = predictionBaseVars["Latitude_0"].values
# plongs = predictionBaseVars["Longitude_0"].values

#         
ilongs[ilongs > 100] -= 360
# plongs[plongs > 100] -= 360

# lats = inversePredictionBaseVars["Latitude_0"].values
# longs = inversePredictionBaseVars["Longitude_0"].values

# Create a map with custom boundaries
plt.figure(figsize=(10, 8))
map = Basemap(projection='cyl', llcrnrlon=mapcorners[0], llcrnrlat=mapcorners[1], urcrnrlon=mapcorners[2], urcrnrlat=mapcorners[3], lat_ts=20, resolution='c',lon_0=135)

# Set the map boundaries
map.drawcoastlines()
map.drawcountries()

# Draw latitude and longitude grid lines
map.drawparallels(range(-90, 91, 30), labels=[1,0,0,0], fontsize=10)  # Latitude lines
map.drawmeridians(range(-180, 181, 45), labels=[0,0,0,1], fontsize=10)  # Longitude lines

# Convert latitude and longitude to map coordinates
ix, iy = map(ilongs, ilats)
# Convert latitude and longitude to map coordinates
# px, py = map(plongs, plats)

map.scatter(ix, iy, s=20, c=variable, cmap='cool', label='PASTEL Modeled Missing Data',edgecolor='none')

plt.colorbar(label='Value of data', shrink=0.6)

# # Plot latitude and longitude from predictionBaseVars dataframe in red
# map.scatter(px, py, s=20, c=y_free_predict, cmap='Wistia_r', label='True ATom Data',edgecolor='none')

# plt.colorbar(label='Concentration of ATom Data ' + unit_label, shrink=0.6)

# Set labels and title
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
plt.title('Data along ATom flight track')


# Show plot
plt.show()