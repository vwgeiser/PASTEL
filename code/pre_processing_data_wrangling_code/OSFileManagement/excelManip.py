# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:02:36 2023

This module contains functions that were used along side the PVOCAL model. 

During trajectory generation 2 errors may occur where this file is necessary. 

1. If an error occurs where trajectory is not found it could just be that it needs to be renamed as files are somtimes get associated with the wrong date (on the edges of meteorology files) causing them to "not exist" in the directory. This is why bulkRename is necessary.

2. Sometimes the initial conditions do not generate a trajectory or a corrupt trajectory is generated. When calculating the meteorology variables on the trajectory path this will become imparative. this casues a need for the bulkMove fuction (to move them to a jail directory) or bulkDelete (remove them from the directory completely).

@author: Victor Geiser
"""

import shutil
import os
        
def bulkDel(wdir, badstring):   
    """
    Function that deletes all files in a directory containing a certain string 

    Parameters
    ----------
    wdir : string
        String to the working directory containing files with "badstring" that will be deleted.
    badstring : string
        If a file contains "badstring" that file will be deleted.

    Returns
    -------
    None.

    """
    for filename in os.listdir(wdir):
        if badstring in filename:
            f = os.path.join(wdir, filename)
            os.remove(f)
            
            
#put in jail
def bulkMoveReg(oldDir, newDir, tstr):
    """
    Funtion that moves all files within a directory containing a certain string to a different directory

    Parameters
    ----------
    oldDir : string
        String to the directory containing files that have the target string ("tstr") in the name.
    newDir : string
        String to the new directory that the foles containing "tstr" should be moved to.
    tstr : string
        Target string for the funtion. files containing this string will be the ones moved.

    Returns
    -------
    None.

    """
    for filename in os.listdir(oldDir):
        if tstr in filename:
            f = os.path.join(oldDir, filename)
            shutil.move(f, newDir)

def bulkRename(wdir, oldStr, newStr):
    """
    Function that remanes all files containting "oldStr" in a directory to a file containing "newStr"

    Parameters
    ----------
    wdir : string
        Working directory to directory with files containing "oldStr".
    oldStr : string
        Target string. Files containing "oldStr" will be renamed.
    newStr : string
        New string. Files that used to contain "oldStr" will now contain "newStr"

    Returns
    -------
    None.

    """
    for filename in os.listdir(wdir):
        if oldStr in filename:
            f = os.path.join(wdir, filename)
            x = f.removesuffix(oldStr)        
            os.rename(f, x + newStr)
