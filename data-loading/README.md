Essa pasta contém os scripts e recursos utilizados para gerar as instâncias de dados de cada uma das cidades. De um modo geral, o objetivo é criar uma tabela final ("tabela_resultante_boston.sql") em que as entradas são geolocalizadas e possuem as features que serão usadas para treinamento e teste do modelo.  

Os arquivos .json são usados no QGIS para coletar os dados dos prédios ao redor dos pontos (ferramenta: "Unir atributos pela localização (sumário)").  

Para que os .sql possam ser executados a extensão "postgis" é necessária. 

Para rodar os scripts python um ambiente virtual deve ser criado a partir de "requirements.txt".  

O banco de dados e a tabela utilizada para cada uma das cidades tem os seguintes nomes:  

- Chicago: tccbase_2 (var_radial_chicago)  

- Boston: tccbase_boston (var_radial)  
 
- Los Angeles: tccbase_la (var_radial)

- Washington: tccbase_dc (var_radial) 

- Manhattan: tccbase (var_radial_buffer_sky_exposure_temp)

### Fonte dos dados

- Chicago: https://data.cityofchicago.org/Buildings/Building-Footprints-current-/hz9b-7nh8

- Boston: https://geodata.mit.edu/catalog/mit-mxfdtdiaufj7u

- Los Angeles: https://geohub.lacity.org/datasets/813fcefde1f64b209103107b26a8909f_0/explore?location=34.0375062C-118.4601232C17.62

- Washington: https://opendata.dc.gov/documents/274f7c2b5f7c4ae19f165d9951057a00/explore 

- Manhattan: https://data.cityofnewyork.us/Housing-Development/Building-Footprints/nqwf-w8eh

### Dados processados

Os dados já processados pelos scripts de "data-loading" podem ser encontrados em:  

https://drive.google.com/drive/folders/1g-Q9o_KXD40kQNjVVI5P5D7E-xI7_fmf?usp=sharing  