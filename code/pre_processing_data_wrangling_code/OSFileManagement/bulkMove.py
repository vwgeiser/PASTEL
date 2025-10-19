# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:06:47 2023

@author: SARP
"""

import shutil
import os

# assign directory
#directory = r"C:\Users\SARP\Documents\NASASARP\trajectories"
directory = r"C:\Users\SARP\Documents\NASASARP\trajectories"

newDir = r"C:\Users\SARP\Documents\NASASARP\trajectoryJAIL"
 
# iterate over files in
# that directory
string = '2022062402'      
        
#put in jail
for filename in os.listdir(directory):
    if string in filename:
        f = os.path.join(directory, filename)
        shutil.move(f, newDir)
        
##----------------------------------------------------------

# #put back in trajectories    
# newDir = r"C:\Users\SARP\Documents\NASASARP\trajectories"

# directory = r"C:\Users\SARP\Documents\NASASARP\trajectoryJAIL"  
      
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     shutil.move(f, newDir)