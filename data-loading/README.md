This folder contains the scripts and resources used to generate the data instances of each city. In general, the goal is to create a final table ("tabela_resultante_boston.sql") in which the entries are geolocated and have the features that are going to be used in training and test.

The files .json are used in QGIS to collect building data around the points (tool: "Join Attributes by Location (summary)")
 
The "postgis" extension is required to execute the .sql files.

To execute the python scripts it is necessary to create a virtual environment from "requirements.txt".

The database and table used for each one of the cities have the following names:

- Chicago: tccbase_2 (var_radial_chicago)  

- Boston: tccbase_boston (var_radial)  
 
- Los Angeles: tccbase_la (var_radial)

- Washington: tccbase_dc (var_radial) 

- Manhattan: tccbase (var_radial_buffer_sky_exposure_temp)

### Data sources

- Chicago: https://data.cityofchicago.org/Buildings/Building-Footprints-current-/hz9b-7nh8

- Boston: https://geodata.mit.edu/catalog/mit-mxfdtdiaufj7u

- Los Angeles: https://geohub.lacity.org/datasets/813fcefde1f64b209103107b26a8909f_0/explore?location=34.0375062C-118.4601232C17.62

- Washington: https://opendata.dc.gov/documents/274f7c2b5f7c4ae19f165d9951057a00/explore 

- Manhattan: https://data.cityofnewyork.us/Housing-Development/Building-Footprints/nqwf-w8eh

### Processed Data

The data already processed by the scripts of "data-loading" can be found at:

https://drive.google.com/drive/folders/1g-Q9o_KXD40kQNjVVI5P5D7E-xI7_fmf?usp=sharing  
