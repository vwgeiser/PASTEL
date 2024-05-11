# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:53:15 2023

This file is used to generate a list of complete URLs to input into the "getMetData" .py file

the directory for these files is hosted here: https://www.ready.noaa.gov/data/archives/gdas1/

our goal is to form a list of all of the urls we need into a list we can loop over.

@author: Victor Geiser
"""


#--------- Getting download urls made ---------#

import pandas as pd

#read in csv
df1 = pd.read_csv(r"C:\Users\vwgei\Documents\PVOCAL\data\ATOM.csv")

#read in just dates 
dates  = df1.iloc[:, 1]

#split the dates from the date and the time
splitDates = []
for date in dates:
    splitDates.append(date.split())

#grab just the date portion
newDates = []
for date in splitDates:
    newDates.append(date[0])

#loop through dates and put any unique dates into a list
uniqueDates = []
for date in newDates:
    if (date not in uniqueDates):
        uniqueDates.append(date)

#print(uniqueDates) #DEBUG


# uncomment if dateTime object is not in string format already
# strDates = []
# for date in uniqueDates:
#     strDates.append(date.strftime("%Y%m%d"))
    
# print(strDates) #DEBUG


"""
NOTE: The following code is for URL creation for GDAS data urls.
"""
#Container for URLs
URLlist = []

#loop through unique dates
for date in uniqueDates:
    #the base of all urls
    base = "https://www.ready.noaa.gov/data/archives/gdas1/gdas1."
    
    #split the dates into strings by using the "/" character
    spdate = date.split("/")
    
    #start of month encoding
    if(spdate[0] == '6'):
        month = 'jun'
    if(spdate[0] == '7'):
        month = 'jul'
    if(spdate[0] == '1'):
        month = 'jan'
    if(spdate[0] == '2'):
        month = 'feb'
    if(spdate[0] == '3'):
        month = 'mar'
    if(spdate[0] == '4'):
        month = 'apr'
    if(spdate[0] == '5'):
        month = 'may'
    if(spdate[0] == '8'):
        month = 'aug'
    if(spdate[0] == '9'):
        month = 'sep'
    if(spdate[0] == '10'):
        month = 'oct'
    if(spdate[0] == '11'):
        month = 'nov'
    if(spdate[0] == '12'):
        month = 'dec' 
        
    #year encoding
    year = spdate[2]    
    
    #start of date encoding
    if((int(spdate[1]) > 1) and (int(spdate[1]) < 8)):
        ext = ".w1"
    if((int(spdate[1]) > 7) and (int(spdate[1]) < 15)):
        ext = ".w2"
    if((int(spdate[1]) > 15) and (int(spdate[1]) < 22)):
        ext = ".w3"
    if((int(spdate[1]) > 22) and (int(spdate[1]) < 29)):
        ext = ".w4"
    if(int(spdate[1]) > 28):
        ext = ".w5"
        
    #Put them all together    
    fullurl = base + month + year + ext
    
    #append to a new list
    URLlist.append(fullurl)

#initiallize a final container for unique dates    
uniqueURLlist = []    

#Loop through and append unique dates
for url in URLlist:
    if (url not in uniqueURLlist):
        uniqueURLlist.append(url)


"""
NOTE: An earlier NAM data implementation is below (currently commented out) that may or may not be operational.
"""
#--------- Getting NAM download urls made ---------#

#pd.DatetimeIndex(dates)
#uniqueDates = []

#put unique dates in a list
#for date in dates:
#    if (date not in uniqueDates):
#        uniqueDates.append(date)

#make them a datetime object
#pd.to_datetime(uniqueDates)

#20090625_nam12 this is the goal form

#strDates = []

#okay but actually make it a string now
#for date in uniqueDates:
#    strDates.append(date.strftime("%Y%m%d"))
    
#print(strDates)

#URLlist = []

#for date in strDates:
#    url = "https://www.ready.noaa.gov/data/archives/nam12/" + date + '_nam12'
#    URLlist.append(url)

#--------BEGIN MY OBSESSIVE PRINT DEBUG STATEMENTS--------#

#print(dates)
#print(uniqueDates)
#print(strDates)
#print(URLlist)

        