## Data visualization pipeline for photogrammetry data.  
Author: Hina Suzuki hsuzuki@freefallaerospace.com [hsuzuki-ff-sys](http://github.com/hsuzuki-ff-sys) / [hina18201716](https://github.com/hina18201716)


### Useage
For flat surface measuremnts with reference bars (e.g. Ring characterizaitons), use vis.ipynb to visualize data and fit flat surface. 

For measurements of curved 2D surfaces (e.g. Radius of Curvature measurements of inflatable optics), run following notebooks in order. 

- #### CleaningData 
    From data with noise and fake points, select points of interests and save in a coordinats defined by points of interests. This will create a new text file that only store selected data points. 
    This notebook read-in raw data, select reference bar points, 
- #### Ring Aligntment 

\
User input and lines to be checked manually is indicated as ''##USER INPUT''. Those lines are planned to be improved. 

### Requirements
- [Python 3.14 +](https://www.python.org/downloads/)
- [numpy](http://pypi.org/project/numpy/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [scipy](https://pypi.org/project/scipy/)
- [jupyter](https://pypi.org/project/jupyter/)

All pachages can be installed by running following.
```bash
pip install -r requirements.txt
```
