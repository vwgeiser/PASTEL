@echo off
echo Starting Python scripts...

python V4_Trajectory_Generator_NAM.py
echo Finished NAM script.

python V4_Trajectory_Generator_GDAS.py
echo Finished GDAS script.

python V4_Trajectory_Generator_NCEP.py
echo Finished NCEP script.

echo All scripts have completed.
pause