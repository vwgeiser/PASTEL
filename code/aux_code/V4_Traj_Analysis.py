import os
import pandas as pd
import numpy as np
import re
import pysplit

# Path to the trajectory files
path = 'C:/Users/vwgei/Documents/PVOCAL/data/AllTraj'
dir_list = os.listdir(path)

# Create a list of full file paths
filelist = [os.path.join(path, file) for file in dir_list]

traj_iindex = []
for file in filelist:
    traj_iindex.append(file.split("[")[1].split(']')[0])

# Convert the strings in traj_iindex to integers and sort them
sorted_traj_iindex = sorted(traj_iindex, key=int)

# Make trajectory group
trajgroup = pysplit.make_trajectorygroup(filelist)

# Initialize a list to collect DataFrames
container = []

# Column names for output DataFrame
column_names = [
    'Timestep', 'Pressure', 'Potential_Temperature', 'Temperature',
    'Rainfall', 'Mixing_Depth', 'Relative_Humidity', 'Specific_Humidity',
    'Mixing_Ratio', 'Terrain_Altitude', 'Solar_Radiation', 'geometry',
    'DateTime', 'Temperature_C', 'Distance_ptp', 'Cumulative_Dist',
    'Dist_from_origin', 'bearings_from_origin', 'bearings_ptp', 'iindex'
]

# Define a function to rename columns based on timestep index
def rename_columns(df, timestep):
    return {col: f"{col}_{timestep}" for col in df.columns}


# Get final path data endpoints from the 0 hour and -24 hour timestamps
for counter, traj in enumerate(trajgroup):
    # print("here")
    print(traj.datetime[0])
    print(traj.filename)
    traj.calculate_distance()
    traj.calculate_vector()
    traj.calculate_moistureflux()
    
    x = traj.data 
    
    transformed_data = pd.DataFrame()

    # Create a list of DataFrames for each timestep
    timestep_dfs = []

    for timestep in range(-24, 1):
        timestep_data = x[x['Timestep'] == timestep].drop(columns='Timestep')
        timestep_data = x[x['Timestep'] == timestep].drop(columns='Timestep')
        timestep_data.rename(columns=rename_columns(timestep_data, timestep), inplace=True)
        
        # Add 'iindex' column
        timestep_data['iindex'] = np.arange(len(timestep_data))
        
        timestep_dfs.append(timestep_data)

    # Concatenate all timestep DataFrames and reset index
    transformed_data = pd.concat(timestep_dfs, ignore_index=True)
    
    # Reorder columns with 'iindex' as the first column
    columns = ['iindex'] + [col for col in transformed_data.columns if col != 'iindex']
    transformed_data = transformed_data[columns]

    # Collapse all data into a single row
    transformed_data = transformed_data.groupby('iindex').first().reset_index()

    # Append the transformed DataFrame to the container
    container.append(transformed_data)

    print(f"finished Traj: {counter} There are {len(filelist) - counter - 1} remaining...")

# Concatenate the list of DataFrames into a single DataFrame
concatenated_df = pd.concat(container, ignore_index=True)

concatenated_df['iindex'] = sorted_traj_iindex

# Save the concatenated DataFrame to a CSV file
concatenated_df.to_csv(r"C:\Users\vwgei\Documents\PVOCAL\data\V4_trajectory_data.csv", index=False)