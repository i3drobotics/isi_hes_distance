# ISI Hes Distance

Measure the distance of point infront deimos camera when mounted to HES probe.

## required modules

 - numpy
 - opencv-python

Command to install modules: python -m pip install numpy opencv-python

## Run

Command to run: 
```bash

python RangeFunctionEditedAndSanityCheck.py

```
Will print the distance to the terminal

## Running from labview

Script: RangeFunctionEditedAndSanityCheck.py

Use the function 'ranger' to get the range of center in front of deimos. (Returns Double: distance)

Use 'sanity' to check for the existance of the laser in the images.
(Returns Boolean: True/False)