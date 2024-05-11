# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:08:31 2023

@author: SARP
"""

import numpy as np
import pandas as pd
#import shapely
#import sklearn.ensamble as RandomForestClassifier
import matplotlib.pyplot as plt 
import pysplit 

from joblib import dump, load

# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap

# Symbolize by trajectory endpoint altitude
def trajColorAlt(altitude):
    if(altitude > 11000): return 'red'
    if(altitude > 10000): return 'orange'
    if(altitude > 9000): return 'olive'
    if(altitude > 8000): return 'green'
    if(altitude > 6000): return 'cyan'
    if(altitude > 4000): return 'blue'
    if(altitude > 3000): return 'purple'
    if(altitude > 2000): return 'pink'
    if(altitude > 1000): return 'gray'
    if(altitude > 0): return 'brown'
    
def symbByMonth(month):
    if(month == 1): return 'maroon'
    if(month == 2): return 'red'
    if(month == 3): return 'coral'
    if(month == 4): return 'orange'
    if(month == 5): return 'gold'
    if(month == 6): return 'yellow'
    if(month == 7): return 'olive'
    if(month == 8): return 'green'
    if(month == 9): return 'cyan'
    if(month == 10): return 'blue'
    if(month == 11): return 'indigo'
    if(month == 12): return 'purple'

trajgroup = pysplit.make_trajectorygroup(r"C:\Users\vwgei\Documents\PVOCAL\data\GDASTrajectories\*")

# #mapcorners = [-155, 20, -110, 52] #all extent


mapcorners = [-260, -90, 20, 90] #full globe
# mapcorners = [-100, -10, 100, 30] #reverse extent

standard_pm = None

bmap_params = pysplit.MapDesign(mapcorners, standard_pm)

bmap = bmap_params.make_basemap()

# # Remove parallels and meridians
# for line in bmap.ax.lines:
#     if line.get_label() == 'parallels' or line.get_label() == 'meridians':
#         line.remove()

# Longitude

# # Draw latitude and longitude grid lines
# bmap.drawparallels(range(-90, 91, 30), labels=[1,0,0,0], fontsize=10)  # Latitude lines
# bmap.drawmeridians(range(-180, 181, 45), labels=[0,0,0,1], fontsize=10)  

for traj in trajgroup:
    altitude0 = traj.data.geometry.apply(lambda p: p.z)[0]
    traj.trajcolor = trajColorAlt(altitude0) #uncomment to symbolize by altitude
    
    # month0 = traj.data.DateTime[0].month
    # traj.trajcolor = symbByMonth(month0)

    
for traj in trajgroup[:]:
    bmap.plot(*traj.path.xy, c=traj.trajcolor, latlon=True, zorder=20)
    
    
    
    
# PVOCAL_EST = load(r"C:\Users\vwgei\Documents\PVOCAL\bestmodels\WAS_DMS_WAS.joblib")
    

# # Read in CSV file with all of the data we need. Meteorology variables + Pathdata + VOC data
# file_path = r"C:\Users\vwgei\Documents\PVOCAL\data\All_Timesteps.csv"

# # Load data from CSV into a pandas DataFrame.
# all_time = pd.read_csv(file_path)

# all_time = all_time.dropna()

# Atom_predict = all_time.iloc[:, np.r_[0:10,12]]

# y_ATom_trajpoint = PVOCAL_EST.predict(Atom_predict)
    
# ATom_traj_data = pd.DataFrame(data=np.column_stack([Atom_predict.values, y_ATom_trajpoint]),columns=Atom_predict.columns.tolist() + ['Modeled_DMS_Concentration'])



# lats = ATom_traj_data["Latitude_0"].values
# longs = ATom_traj_data["Longitude_0"].values
# altitude = ATom_traj_data["AltP_meters_0"].values
# concentration = ATom_traj_data['Modeled_DMS_Concentration'].values

# # Create a map
# plt.figure(figsize=(10, 8))
# mapcorners = [-260, -90, 20, 90] #full globe
# # mapcorners = [-100, -10, 100, 30] #reverse extent

# standard_pm = None

# bmap_params = pysplit.MapDesign(mapcorners, standard_pm)

# bmap = bmap_params.make_basemap()

# # Draw coastlines, countries, and states
# m.drawcoastlines()
# m.drawcountries()
# m.drawstates()

# # Convert latitude and longitude to map coordinates
# x, y = bmap(longs, lats)

# # Set the size of points based on altitude
# sizes = [alt * .05 for alt in altitude]  # Adjust multiplier to your preference

# # Plot points
# bmap.scatter(x, y, s=sizes, c=concentration, cmap='viridis', alpha=0.7, edgecolor='k')

# # Add colorbar
# plt.colorbar(label='Concentration')

# plt.title('Concentration vs Location')
# plt.show()

# #zoomed view    
# mapcorners =  [-125, 32, -115, 43]
# standard_pm = None

# bmap_params = pysplit.MapDesign(mapcorners, standard_pm)

# bmap = bmap_params.make_basemap()

# for traj in trajgroup:
#     altitude0 = traj.data.geometry.apply(lambda p: p.z)[0]
#     #traj.trajcolor = trajColorAlt(altitude0)

# count = 0    
    
# for traj in trajgroup[:]:
#     bmap.plot(*traj.path.xy, c=traj.trajcolor, latlon=True, zorder=20)
#     count += 1
