import pandas as pd

from utils import mongo_db
from utils import dataset

success_in_connection, conn, conn_info = mongo_db.open_mongo_secure_connection()
if success_in_connection:
    print('Yup you are connected')
    print(conn_info)

training_df = dataset.get_raw_data_as_df(
    mongo_conn=conn,
    training_filter={"Lge": "GER1", "Sea": "05-06"}
)  