import pandas as pd

from utils import mongo_db
from utils import dataset

from models.imputation.models import RollingAverage

pd.options.display.max_columns = None

s, mongo_connection, msg = mongo_db.open_mongo_secure_connection()
df = dataset.get_raw_data_as_df(
    mongo_conn=mongo_connection, training_filter={"Sea": "18-19", "Lge": "SPA1"}
)

new_df = RollingAverage(input_to_new_cols=True, window_size=10).impute(df)
new_df[(new_df["HT"] == "Real Madrid") | (new_df["AT"] == "Real Madrid")]
