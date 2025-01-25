# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:33:38 2023

@author: SARP
"""

import pandas as pd
import numpy as np

# def splitPointZ(file):
#     df = pd.read_csv(file)
#     new_columns = {}
    
#     for column in df.columns:
#         if 'geometry_' in column:
#             coords = []
#             longitude = []
#             latitude = []
#             altitude = []
#             for point_Str in df[column]:
#                 if not pd.isna(point_Str) and point_Str.strip() != '':
#                     point_str = str(point_Str)[9:-1]
#                     splitted = point_str.split(" ")
#                     floatSplit = map(float, splitted)
#                     floatList = list(floatSplit)
#                     coords.append(floatList)
#                 else:
#                     coords.append([np.nan, np.nan, np.nan])  # Append NaN values for missing entries
            
#             for co in coords:
#                 longitude.append(co[0])
#                 latitude.append(co[1])
#                 altitude.append(co[2])
            
#             new_column_names = ['Longitude_' + column.split('_')[-1], 
#                                'Latitude_' + column.split('_')[-1], 
#                                'AltP_meters_' + column.split('_')[-1]]
            
#             new_columns[column] = new_column_names
            
#     # Reorder columns based on original geometry columns
#     for old_column, new_column_names in new_columns.items():
#         original_position = df.columns.get_loc(old_column)
#         for i, new_column in enumerate(new_column_names):
#             if new_column not in df.columns:  # Check if column already exists
#                 df.insert(original_position + i + 1, new_column, df[new_column])
#         # Drop the original geometry column
#         df.drop(columns=old_column, inplace=True)
            
#     df.to_csv(file, index=False)



def splitPointZ(file):
    df = pd.read_csv(file)
    
    for column in df.columns:
        if 'geometry_' in column:
            coords = []
            longitude = []
            latitude = []
            altitude = []
            for point_Str in df[column]:
                if not pd.isna(point_Str) and point_Str.strip() != '':
                    point_str = str(point_Str)[9:-1]
                    splitted = point_str.split(" ")
                    floatSplit = map(float, splitted)
                    floatList = list(floatSplit)
                    coords.append(floatList)
                else:
                    coords.append([np.nan, np.nan, np.nan]) 
            
            for co in coords:
                longitude.append(co[0])
                latitude.append(co[1])
                altitude.append(co[2])
            
            new_column_names = ['Longitude_' + column.split('_')[1], 
                                'Latitude_' + column.split('_')[1], 
                                'AltP_meters_' + column.split('_')[1]]
            
            df[new_column_names[0]] = longitude
            df[new_column_names[1]] = latitude
            df[new_column_names[2]] = altitude
            
        
            
    df.to_csv(file, index=False)

filename = r"C:\Users\vwgei\Documents\PVOCAL\data\V4\V4_Awakens.csv"
splitPointZ(filename)


