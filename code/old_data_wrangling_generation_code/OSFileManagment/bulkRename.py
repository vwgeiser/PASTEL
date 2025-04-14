# # -*- coding: utf-8 -*-
# """
# Created on Wed Jul 19 14:25:29 2023

# @author: SARP
# """

# import os

# # assign directory
# directory = r"C:\Users\SARP\Documents\NASASARP\trajectories"
 
# # iterate over files in
# # that directory
# string = '2010070202'
# rstring = '2010070102'
        
# for filename in os.listdir(directory):
#     if string in filename:
#         f = os.path.join(directory, filename)
#         x = f.removesuffix(string)        
#         os.rename(f, x + '2010070102')

import os
import glob

# Define the directory containing the files
directory = r'C:\Users\vwgei\Documents\PVOCAL\data\ERA5Trajectories_backup'

# Use glob to find all files in the directory
files = glob.glob(os.path.join(directory, 'PVOCALERA5*'))

# Iterate through the files and rename them
for file_path in files:
    # Extract the file name
    file_name = os.path.basename(file_path)
    # Replace "PVOCALERA5" with "PASTEL"
    new_file_name = file_name.replace("PVOCALERA5", "PASTEL")
    # Construct the full path for the new file name
    new_file_path = os.path.join(directory, new_file_name)
    # Rename the file
    os.rename(file_path, new_file_path)
    print(f'Renamed: {file_name} -> {new_file_name}')

print("Renaming completed!")
