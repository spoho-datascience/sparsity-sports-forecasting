import pandas as pd

from models.imputation.simple import SimpleImputation
from utils import mongo_db, dataset

success_in_connection, conn, conn_info = mongo_db.open_mongo_secure_connection()
if success_in_connection:
    print('Yup you are connected')
    print(conn_info)

cursor_data = dataset.get_raw_data(
    mongo_conn=conn,
    training_filter={"Sea": {"$in": ["15-16", "16-17", "17-18", "18-19", "20-21"]}},
    extra_sources=['fbd', 'tfmkt_metrics'],
    columns=dataset.TRAINING_COLUMNS + \
        [f'fbd.{c}' for c in [
            "avg_home_odds", "avg_draw_odds", "avg_away_odds",
            "HTS", "ATS", "HST", "AST", "HF", "AF", "HY", "AY", "HR", "AR"
        ]] + \
        [f'tfmkt_metrics.{c}' for c in ["home_starter_total", "away_starter_total"]],
    flatten=True
)

data_df = pd.DataFrame(list(cursor_data))
imputer = SimpleImputation()

print(data_df.isnull().sum())
new_data_df = imputer.impute(data_df)
print(new_data_df.isnull().sum())

imputer.impute(data_df, in_place=True)
print(data_df.isnull().sum())

