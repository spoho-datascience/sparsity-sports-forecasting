import pandas as pd

from utils import mongo_db
from utils import dataset

pd.options.display.max_columns = None

from pymongo.operations import ReplaceOne, UpdateOne

s, mongo_connection, msg = mongo_db.open_mongo_secure_connection()
training_df = dataset.get_raw_data_as_df(
    mongo_conn=mongo_connection, training_filter={}
)
# TFMKT
data = dataset.get_raw_data(
    mongo_connection,
    columns=dataset.TRAINING_COLUMNS
    + [f"tfmkt_metrics.{c}" for c in ["home_starter_total", "away_starter_total"]],
    extra_sources=["tfmkt_metrics"],
    flatten=True,
)

complete_df = pd.DataFrame(data)
complete_df.loc[
    complete_df.duplicated(keep=False, subset=["unique_id"]), ["Sea", "Lge"]
].value_counts()
complete_df["training_result"] = (
    complete_df["HS"].astype(str) + ":" + complete_df["AS"].astype(str)
)
errors_df = complete_df[complete_df["training_result"] != complete_df["result"]]
errors_df

# FOOTBALLDATA
data = dataset.get_raw_data(
    mongo_connection,
    columns=dataset.TRAINING_COLUMNS + [f"fbd.{c}" for c in ["FTHG", "FTAG"]],
    extra_sources=["fbd"],
    flatten=True,
)

complete_df = pd.DataFrame(data).dropna()
errors_df = complete_df[
    (complete_df["HS"].astype(float) != complete_df["FTHG"])
    | (complete_df["AS"].astype(float) != complete_df["FTAG"])
]

# OP
data = dataset.get_raw_data(
    mongo_connection,
    columns=dataset.TRAINING_COLUMNS
    + [
        f"odds_portal.{c}"
        for c in ["home_team", "away_team", "home_score", "away_score"]
    ],
    extra_sources=["odds_portal"],
    flatten=True,
)

complete_df = pd.DataFrame(data).dropna()

errors_df = complete_df[
    (complete_df["HS"].astype(float) != complete_df["home_score"])
    | (complete_df["AS"].astype(float) != complete_df["away_score"])
]
errors_df
errors_df.to_csv("errors.csv")


# Next
data = dataset.get_prediction_set_as_df(mongo_connection)
complete_df = pd.DataFrame(data)

# Duplicates
list_of_duplicates = [
    145868,
    145879,
    145894,
    145896,
    145907,
    145923,
    145939,
    145953,
    145961,
    145974,
    145982,
    145992,
    145999,
    146004,
    146018,
    146036,
    146043,
    146049,
    146055,
    146058,
    146074,
    146083,
    160217,
    160233,
    160242,
    160252,
    160273,
    160283,
    160303,
    160323,
    160343,
    160374,
    160382,
    160411,
    160433,
    160452,
    160503,
    160553,
    160574,
    191417,
    191425,
    191435,
    191438,
    191453,
    191458,
    191465,
    191474,
    191486,
    191496,
    191507,
    191515,
    191529,
    191540,
    191550,
    191559,
    191564,
    191570,
    191576,
    191584,
    191593,
    191598,
    191607,
    191616,
    191628,
    191650,
    191659,
    191662,
    191670,
    191679,
    191689,
    191697,
    191706,
    191714,
    205865,
    205874,
    205884,
    205894,
    205902,
    205911,
    205920,
    205930,
    205939,
    205948,
    205956,
    205966,
    205975,
    205983,
    205989,
    206000,
    206010,
    206017,
    206024,
    206029,
    206041,
    206046,
    206059,
    206073,
    206083,
    206092,
    206099,
    206109,
    206119,
    206126,
    206134,
    206142,
    206153,
    206162,
    292413,
    292418,
    292439,
    292447,
    292454,
    292460,
    292468,
    292478,
    292485,
    292490,
    292495,
    292505,
    292516,
    292520,
    292528,
    292531,
]

db_collection = mongo_connection.get_database("soccer-challenge").get_collection("t")
db_collection.update_many(
    {"training_id": {"$in": list_of_duplicates}},
    {"$unset": {"training_id": ""}},
)

# new
data = dataset.get_raw_data(
    mongo_connection,
    training_filter={
        "Sea": {"$in": ['15-16', '16-17', '17-18', '18-19', '20-21', '21-22', '22-23']},
        "Lge": {"$in": ['SPA1', 'ITA1', 'FRA1', 'ENG1']}
    },
    columns=[c for c in dataset.TRAINING_COLUMNS if c not in ['iso_date']]
    + [f"tfmkt.{c}" for c in dataset.SOURCES["tfmkt"]['columns']]
    + [f"tfmkt_metrics.{c}" for c in ["home_starter_total", "away_starter_total"]]
    + [f"events_info.{c}" for c in ["game_id", "event_num", "type", "minute", "extra", "action", "club", "player_href"]],
    extra_sources=["tfmkt", "tfmkt_metrics"],
    flatten=False,
)
