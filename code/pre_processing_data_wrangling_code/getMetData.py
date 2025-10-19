# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:30:36 2023

@author: Victor Geiser
"""
#import our packages
import wget
import time

urlList = ['https://www.ready.noaa.gov/data/archives/gdas1/gdas1.may12.w1', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.may12.w2',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.may12.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.may12.w4',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.may12.w5',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun12.w1', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun12.w2',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun12.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun12.w4',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun12.w5', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul14.w1'
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul14.w2', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul14.w3',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul14.w4', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul14.w5',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.aug14.w1', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.aug14.w2',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul19.w1', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul19.w2',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul19.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul19.w4',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jul19.w5',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.aug19.w1', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.aug19.w2', 
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.aug19.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.aug19.w4',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.aug19.w5',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.sep19.w1', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.sep19.w2',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.apr16.w1'
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.apr16.w2', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.apr16.w3',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.apr16.w4', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.apr16.w5', 
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.may16.w1', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.may16.w2',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.may16.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.may16.w4',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.may16.w5', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun16.w1',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun16.w2', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun16.w3',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.jun16.w4', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.aug13.w1',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.aug13.w2', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.aug13.w3',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.aug13.w4', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.aug13.w5',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.sep13.w1', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.sep13.w2',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.sep13.w3', 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.sep13.w4',
 'https://www.ready.noaa.gov/data/archives/gdas1/gdas1.sep13.w5']


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