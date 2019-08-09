# python_tools
Collection of useful python scripts for 3D

# Generate3D
Generate point cloud and disparity map from stereo images
## required modules
 - numpy
 - matplotlib
 - opencv-python
 - pptk

Command to install modules: python -m pip install numpy matplotlib opencv-python pptk

## run
Command to run: python generate3D.py
default settings:
- loads images from 'input/'
- loads calibration xml files from 'cal/'
- outputs disparity maps to 'output/disparity'
- outputs point clouds to 'output/point_clouds'
- matcher settings can be changed inside generate3D.py
If pose of camera is known the point clouds can be transformed.
Pose should be provided in the input folder with the pose in format [x,y,z,w,x,y,z].
Pose file should have extension '.txt' however this wildcard can be adjusted
