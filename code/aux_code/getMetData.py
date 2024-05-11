# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:30:36 2023

@author: Victor Geiser
"""
#import our packages
import wget
import time

urlList = ['https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul09.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul09.w4', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun10.w4', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun10.w5', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul10.w5', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun11.w4', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun11.w5', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun12.w4', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun13.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun14.w4', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun15.w4', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun16.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun17.w4', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun18.w4', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul19.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.dec21.w1', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.dec21.w2', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul09.w2', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun10.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul10.w5', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun10.w5', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul10.w4', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun12.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun13.w2', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun14.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun15.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun16.w2', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun17.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun18.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul19.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.nov21.w5', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun11.w5', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun10.w2', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun09.w5', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun09.w2', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun10.w2', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun10.w1', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun10.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.may22.w5']


filesLeft = len(urlList)

#basic timer start
tic = time.perf_counter()

#downloads files from the NOAA hysplit FPT server to the working directory
#with a basic timer for my own curiousity record: 18432!
for url in urlList:
    print("downloading...")
    filename = wget.download(url)
    filesLeft = filesLeft - 1
    print("DONE. There are ", filesLeft, " files remaining.")
    
#basic timer end  
toc = time.perf_counter()

print("Total elasped time is: ")
print(toc - tic)