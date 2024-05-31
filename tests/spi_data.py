import csv
import os

import pandas as pd

path_to_file = os.path.join('input', 'spi_data', 'spi_matches.csv')
spi_df = pd.read_csv(path_to_file)

translator_dict = {}
for row in spi_df[['league_id', 'league']].to_dict(orient='records'):
    translator_dict[row['league_id']] = row['league']

spi_to_training = {
    1843: 'FRA1', 
    2411: 'ENG1', 
    1869: 'SPA1', 
    1854: 'ITA1', 
    1845: 'GER1',
    1975: 'MEX1', 
    1951: 'USA1', 
    1874: 'SWE1', 
    1859: 'NOR1', 
    2105: 'BRA1', 
    1866: 'RUS1', 
    1952: 'MEX1', 
    1827: 'AUT1', 
    1879: 'CHE1', 
    1844: 'FRA2', 
    1846: 'GER2', 
    2412: 'ENG2', 
    2417: 'SCO1', 
    1864: 'POR1', 
    1849: 'HOL1', 
    1871: 'SPA2', 
    1856: 'ITA2', 
    5641: 'ARG1', 
    2160: 'USA2', 
    1837: 'DNK1', 
    1832: 'BEL1', 
    1979: 'CHN1', 
    1947: 'JPN1', 
    2413: 'ENG3', 
    1983: 'ZAF1', 
    2414: 'ENG4', 
    1884: 'GRE1', 
    1948: 'AUS1', 
}

spi_df['TRAINING_COMPETITION'] = spi_df['league_id'].replace(spi_to_training)


import pandas as pd
from utils import mongo_db
from utils import dataset

from scripts import metadata_ops

success_in_connection, conn, conn_info = mongo_db.open_mongo_secure_connection()
if success_in_connection:
    print('Yup you are connected')
    print(conn_info)

dataset.get_schema(conn, 'spi_matches')['Key'].unique()

data = dataset.get_raw_data(
    conn,
    training_filter={"Lge": "GER1"},
    columns=dataset.TRAINING_COLUMNS + [f"spi.{c}" for c in dataset.SOURCES['spi']['columns']],
    extra_sources=['spi'],
    flatten=True
)

df = pd.DataFrame(data)
df

