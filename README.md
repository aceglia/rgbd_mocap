
<p align="center">
    RGBDMocap
</p>

`RGBDMocap` is a Python package that provide accurate motion capture system based on a single RGBD camera with or without skin markers.

# Table of Contents 
[How to install](#how-to-install)

[How to use](#how-to-use)

# How to install
## Installing from anaconda or PIP
We are working on it... Please consider installing from the source

## Installing from the source
If installing from the sources, you will need to install the following dependencies from conda (in that particular order):
- [Python](https://www.python.org/)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [setuptools](https://pypi.org/project/setuptools/)
- [biorbd](https://github.com/pyomeca/biorbd) (optional: for musculoskeletal models)
- [bioviz](https://github.com/pyomeca/bioviz) (optional: for skeletal models visualization)

If you want to use it with DeepLabCut please follow the instruction provided on there documentation. 
Please consider using two separate environment, one for model training and the other for run the live instance of DLC.


After you can install the package by running the following command in the root folder of the package:
```bash
pip install .
```
# How to use
For now it is not possible to run the markers tracking algorithm live. The images will need to be prerecorded by following the instructions: 
## Marker-based Mocap
### Markers recording
The method was design to work with white markers with a size of around 2x2cm. 
After markers been placed on the strategical positions,
you can start recording using the python code "rgbd_recording.py" inside the "rgbd_mocap" directory. 

### Initialization
Once images recorded you can initialize the tracking algorithm using the provided GUI by running the "run_GUI.py" file. 
- First select one or more area on the image. It will open tabs for each the parameters need pto be defined. Please not that you can save the crop area that you just selected.
- Several parameters are available to adjust detection please try some configurations to find the most suitable. You can save the selected parameters and/or load previous saved parameters.
- After setting all tabs go to the markers initialization window to initialize the position of each marker on the first frame. 

Do not forget to save your configuration. 

### Markers tracking
once you have initialized the tracking you can go to the "label_markers.py" file. There is several tracking options to consider. 
You can also use a 3D rigid body model but you will have to define the bodies and related markers by following the example provided. 

## Markerless Mocap 
Coming soon...

## Biomechanical analysis
The processing_data module provide several tools to analyze the motion capture data. 
It include tools to compute the joint angles, velocity, acceleration, and center of mass. 
It also provide tools to compute the joint torques and muscle forces (through static optimization). 
Please check out the example "run_biomechanical_analysis.py" for examples of how to use the tools.

## Issues
Please report any issues or bugs you find on the GitHub issue tracker.
If you have any questions or need help, please contact us at amedeo.ceglia@umontreal.ca.
