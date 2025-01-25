# -*- coding: utf-8 -*-
"""
Created on Saturday 10/5/2024 14:32:42

@author: Victor Geiser
Handels all the Preprocessing and trajectory generation before running the main PASTEL script!
"""


# To do - Compare to GMI - PDP
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import time
import wget
import pysplit
import os
import shutil



file_paths = [
r"D:\PVOCAL\Other_Campaigns\ACCLIP.csv",
# r"D:\PVOCAL\Other_Campaigns\DC3.csv",
# r"D:\PVOCAL\Other_Campaigns\DISCOVERAQ.csv",
# r"D:\PVOCAL\Other_Campaigns\FIREXAQ.csv",
# r"D:\PVOCAL\Other_Campaigns\INTEXA.csv",
r"D:\PVOCAL\Other_Campaigns\INTEXB_C130.csv",
r"D:\PVOCAL\Other_Campaigns\INTEXB_DC8.csv",
r"D:\PVOCAL\Other_Campaigns\KORUSAQ.csv",
# r"D:\PVOCAL\Other_Campaigns\PEMTA_DC8.csv",
# r"D:\PVOCAL\Other_Campaigns\PEMTA_P3.csv",
# r"D:\PVOCAL\Other_Campaigns\PEMTB_DC8.csv",
# r"D:\PVOCAL\Other_Campaigns\PEMTB_P3.csv",
# r"D:\PVOCAL\Other_Campaigns\PEMWA.csv",
# r"D:\PVOCAL\Other_Campaigns\PEMWB.csv",
# r"D:\PVOCAL\Other_Campaigns\SEAC4RS_DC8.csv",
# r"D:\PVOCAL\Other_Campaigns\TRACEA_DC8.csv",
# r"D:\PVOCAL\Other_Campaigns\TRACEA_P3.csv",
# r"D:\PVOCAL\Other_Campaigns\TRACEP_DC8.csv",
# r"D:\PVOCAL\Other_Campaigns\TRACEP_P3.csv",
# r"D:\PVOCAL\Other_Campaigns\WINTER.csv",
]

meteo_labels =[
"GDAS", # ACCLIP
# "NAM", # DC3
# "NAM", # DISCOVERAQ
# "NAM", # FIREXAQ
# "NCEP", # INTEXA
"GDAS", # INTEXB C130
"GDAS", # INTEXB DC8
"GDAS", # KORUSAQ
# "NCEP", # PEMTA DC8
# "NCEP", # PEMTA P3
# "NCEP", # PEMTB DC8
# "NCEP", # PEMTB P3
# "NCEP", # PEMWA
# "NCEP", # PEMWB
# "NAM", # SEAC4RS
# "NCEP", # TRACEA DC8
# "NCEP", # TRACEA P3
# "NCEP", # TRACEP DC8
# "NCEP", # TRACEP P3
# "NAM" # WINTER
]

j = 0
for file in file_paths:
    data = pd.read_csv(file)
    print(f"Current file: {file}")

    # Manually create the `DateTime` column
    data['DateTime'] = pd.to_datetime(
        data['Year'].astype(str) + '-' +
        data['Month'].astype(str).str.zfill(2) + '-' +
        data['Day'].astype(str).str.zfill(2) + ' ' +
        data['Hour'].astype(str).str.zfill(2) + ':00:00',
        format='%Y-%m-%d %H:%M:%S'
    )

    label = meteo_labels[j]

    # Replace WAS nodata value with np.nan for consistency
    data.replace(-999999.0, np.nan, inplace=True)
    data.replace(-888888.0, np.nan, inplace=True)
    data.replace(-9.999999e+09, np.nan, inplace=True)
    data.replace(-9.999990e+08, np.nan, inplace=True)
    data.replace(-8.888888e+06, np.nan, inplace=True)
    data.replace(-9.999999e+06, np.nan, inplace=True)

    # Replace WAS nodata value with np.nan for consistency
    data.replace(-8888, np.nan, inplace=True)


    data = data.dropna(subset = "LAT")
    data = data.dropna(subset = "LON")
    data = data.dropna(subset="ALT")

    dataLength = len(data)

    # Make sure these are correct and updated!
    # Read in Lats and Longs
    lat = data['LAT']
    # print(lat)
    long = data['LON']
    # print(long)

    #lat-long tuples
    # LL = list(zip(lat,long))
    data['LL'] = list(zip(lat,long))

    # # Assuming df is your DataFrame and DateTime is the column containing datetime information
    # data['time'] = pd.to_datetime(data['DateTime'])

    # # Create new columns
    # data['Year'] = data['time'].dt.year
    # data['Month'] = data['time'].dt.month
    # data['Day'] = data['time'].dt.day
    # data['Hour'] = data['time'].dt.hour

    # Directory containers for hysplit and where our output files will be
    working_dir = 'C:/hysplit/working'
    storage_dir = f"C:/Users/vwgei/Documents/PVOCAL/data/{label}Trajectories"
    meteo_dir = f'D:/PVOCAL/{label}'

    # Basename shared by all trajectories
    basename = f'PASTEL'
                    
    """
    *****NOTE*****
    HYSPLIT is capable of calculating meteorlogical variables along the trajctory 
    path. I recommend you do this for this model. To do this after you have 
    HYSPLIT on your local machine: 
        
    THIS REQUIRES A LOCAL INSTALLATION OF HYSPLIT (either registered or unregistered version - the unregistered version was used for the inital project)

    1. Run HySPLIT
    2. click "Menu"
    3. click "Advanced"->"Configuration settings"->"Trajectory"
    4. clik "Menu" FOR (6) "Add METEOROLOGY output along trajectory"
    5. select all variables (all is recommended if computation time isn't a huge factor')
    6. CLICK SAVE
    7. CLICK SAVE AGAIN
    8. File size for each trajectory should be about 5 kilobites as opposed to 3 (for 24 hour back trajecories)
    """

    """
    This is the main function PySPLIT is used for. 

    There is however one difference between this function and the generate_bulktraj
    that comes with pysplit as of pysplit version=0.3.5

    This is the inclusion of the iindex parameter is special for this program and 
    my model in that the data I was working with had no index or unique ID to match
    with accross multiple iterations of excel joining and other shenanigans.

    This might not be necessary for your project but it was for mine as some entire
    rows needed to be thrown out.

    There could also be some way that this is done in PySPLIT or a better way
    to generate a usible index not in this form but this is the best way that I 
    figured out for the time/problems I was having.

    ...

    The changes to trajectory_generator.py within pysplit are as follows:    
        
    # Change at line 11 in trajectory_generator.py
    def generate_bulktraj(basename, hysplit_working, output_dir, meteo_dir, years,
                        months, hours, altitudes, coordinates, run, iindex,
                        meteoyr_2digits=True, outputyr_2digits=False,
                        monthslice=slice(0, 32, 1), meteo_bookends=([4, 5], [1]),
                        get_reverse=False, get_clipped=False,
                        hysplit="C:\\hysplit4\\exec\\hyts_std"):
    # NOTE: notice the new "iindex" parameter


    # Change at line 150 to trajectory_generator.py    
    # Add trajectory index and altitude to basename to create unique name # INDEX CHANGE HERE
    trajname = (basename + str(iindex) + m_str  + season + 
                fnameyr + "{0:02}{1:02}{2:02}".format(m, d, h))
    # NOTE: notice once again the inclusion of the "str(iindex)"   
    """

    # data.to_csv(r"C:\Users\vwgei\Documents\PVOCAL\data\TrajectoryGeneratorInput.csv",index=False)

               

    for ind in data.index:
        # print(data["DateTime"][ind], data["Year"][ind], data["Month"][ind], data["Hour"][ind])#, data["MMS_G_ALT"][ind], data["LL"][ind], data["iindex"][ind], "monthslice", slice(data["Day"][ind] - 1, data["Day"][ind], 1)) #DEBUG
        pysplit.generate_bulktraj(basename, 
                                working_dir, 
                                storage_dir, 
                                meteo_dir, 
                                years = [data["Year"][ind]], 
                                months = [data["Month"][ind]], 
                                hours = [data["Hour"][ind]], 
                                altitudes = [data["ALT"][ind]], 
                                coordinates = data["LL"][ind], 
                                run = (-24),
                                iindex = [data["iindex"][ind]],
                                meteoyr_2digits = True,
                                outputyr_2digits = False,
                                monthslice = slice(int(data["Day"][ind]) - 1, int(data["Day"][ind]), 1),
                                meteo_bookends = ([4,5],[1]),
                                get_reverse = False,
                                get_clipped = False,
                                hysplit = "C:/hysplit/exec/hyts_std"
                                )
        print("Trajectory generated for: " + str(ind))
    
    j += 1