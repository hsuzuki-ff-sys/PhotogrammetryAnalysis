# Data Visualization Pipeline for Photogrammetry Data  
Author: Hina Suzuki hsuzuki@freefallaerospace.com 

[hsuzuki-ff-sys](http://github.com/hsuzuki-ff-sys) / [hina18201716](https://github.com/hina18201716)


## Usage
### 1. Ring Flatness Characterization
 For flat surface measuremnts with reference bars (e.g. Ring characterizaitons), use vis.ipynb to visualize data and fit flat surface. 

### 2. Surface Mearuremnts and Fitting  
 For measurements of curved 2D surfaces (e.g. Radius of Curvature measurements of inflatable optics), run following notebooks in order. 

- #### CleaningData 
    From data with noise and fake points, select points of interests and save in a coordinats defined by points of interests. This will create a new text file that only store selected data points. 
    This notebook read-in raw data, select reference bar points, rotate based on SVD from selected data, selection of points of interest. Use rotation and cropping iteratively to get clean data. 
    
- #### RingAligntment 
    Calibrate coordinates s.t. ring points align over each data sets. User manually select indecies of three point pairs to be calibrated against. 

- #### PressureSweep
    Consolidate data from multiple pressures and apply surface fit (cubic/even polynomial) on each data. Vizualize xz, yz cut of the surface. 

User input and lines to be checked manually is indicated as **#USER INPUT**. See examples branch for examples. 

## Design 
We design this pipeline so it make easier f8or user to keep track of data. We define following data stage. 

- ``` Data/ ``` This directory hold data directly from [V-STARS](https://www.geodetic.com/v-stars/) from Geodetic Systems software and other cleaned data in ```.txt``` files.  

- ```figs/``` This stores outouts from pipeline such as plots, that are generated from data in ```Data/```. Those plots should be presentable that can be used to deliver results from experiments. 

## Requirements
- [Python 3.14 +](https://www.python.org/downloads/)
- [numpy](http://pypi.org/project/numpy/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [scipy](https://pypi.org/project/scipy/)
- [jupyter](https://pypi.org/project/jupyter/)

All pachages can be installed by running following.
```bash
pip install -r requirements.txt
```
