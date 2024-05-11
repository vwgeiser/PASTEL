# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:53:15 2023

This file is used to generate a list of complete URLs to input into the "getMetData" .py file

the directory for these files is hosted here: https://www.ready.noaa.gov/data/archives/gdas1/

our goal is to form a list of all of the urls we need into a list we can loop over.

@author: Victor Geiser
"""

#import our packages
import wget
import time

#--------- Getting download urls made ---------#

import pandas as pd

# Assuming 'your_csv_file.csv' is the name of your CSV file and 'date_column' is the name of the column containing datetime objects
df = pd.read_csv(r"C:\Users\vwgei\Documents\PVOCAL\data\PVOCAL_data_wtime.csv")
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Create a set to store unique dates
unique_dates = set()

# Iterate through the rows of the DataFrame
for index, row in df.iterrows():
    # Extract the date in the format 'YYYYMMDD'
    date_str = row['DateTime'].strftime('%Y%m%d')
    
    # Check if the date is not in the set, then add it along with the previous day
    if date_str not in unique_dates:
        unique_dates.add(date_str)
        
        # Add the previous day
        previous_date = (row['DateTime'] - pd.Timedelta(days=1)).strftime('%Y%m%d')
        unique_dates.add(previous_date)

# Transform the set into a list and sort it
final_unique_dates = sorted(list(unique_dates))

# Add the '_nam12' suffix to each date
final_unique_dates = ['https://www.ready.noaa.gov/data/archives/nam12/' + date + '_nam12' for date in final_unique_dates]

# Now, final_unique_dates contains the desired list of unique dates with the specified format
# print(final_unique_dates)

filesLeft = len(final_unique_dates)

#basic timer start
tic = time.perf_counter()

#downloads files from the NOAA hysplit FPT server to the working directory
#with a basic timer for my own curiousity record: 18432!
for url in final_unique_dates:
    print("downloading...")
    filename = wget.download(url)
    filesLeft = filesLeft - 1
    print("DONE. There are ", filesLeft, " files remaining.")
    
#basic timer end  
toc = time.perf_counter()

print("Total elasped time is: ")
print(toc - tic)


        