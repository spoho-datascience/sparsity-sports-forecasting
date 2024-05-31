from utils import mongo_db
from utils import dataset

from pymongo.operations import ReplaceOne, UpdateOne

s, mongo_connection, msg = mongo_db.open_mongo_secure_connection()
raw_data = dataset.get_raw_data(mongo_connection, training_filter={"Lge": "SPA1", "Sea": "21-22"})

def get_map(prefix, from_collection, let_clause, pipe_expr, project_clause, set_clause):
    new_map = [
        # I translated the name of the hometeam to the name I need for football data records using the Teams collection
        {"$lookup": {
            "from": from_collection,
            # Store this aux variables to represent the key of the team (Season, League, Team)
            "let": let_clause,
            "pipeline": [
                {"$match": { 
                    "$expr": pipe_expr
                }},
                {"$project": project_clause}
            ],
            "as": f"{prefix}_mapping"
        }},
        # We don't want the result as an array
        {"$unwind": f"${prefix}_mapping"},
        {"$set": set_clause},
        {"$unset": f"{prefix}_mapping"}
    ]
    return new_map

map_HT_fbd = get_map(
    prefix="HT",
    from_collection="teams_by_season",
    let_clause={"mapping_season": "$Sea", "mapping_league": "$Lge", "mapping_team": "$HT"},
    pipe_expr={ 
        "$and": [
                { "$eq": [ "$Sea",  "$$mapping_season" ] },
                { "$eq": [ "$Lge",  "$$mapping_league" ] },
                { "$eq": [ "$team_train",  "$$mapping_team" ] },
        ]
    },
    project_clause={"team_fbd": 1, "_id": 0},
    set_clause={"HT_fbd": "$HT_mapping.team_fbd"}
)

map_AT_fbd = get_map(
    prefix="AT",
    from_collection="teams_by_season",
    let_clause={"mapping_season": "$Sea", "mapping_league": "$Lge", "mapping_team": "$AT"},
    pipe_expr={ 
        "$and": [
                { "$eq": [ "$Sea",  "$$mapping_season" ] },
                { "$eq": [ "$Lge",  "$$mapping_league" ] },
                { "$eq": [ "$team_train",  "$$mapping_team" ] },
        ]
    },
    project_clause={"team_fbd": 1, "_id": 0},
    set_clause={"AT_fbd": "$AT_mapping.team_fbd"}
)

map_HT_tfmkt = get_map(
    prefix="HT",
    from_collection="teams",
    let_clause={"mapping_team": "$HT"},
    pipe_expr={ "$eq": [ "$team_train",  "$$mapping_team" ] },
    project_clause={"tfmkt": 1, "_id": 0},
    set_clause={"HT_tfmkt": "$HT_mapping.tfmkt"}
)

map_AT_tfmkt = get_map(
    prefix="AT",
    from_collection="teams",
    let_clause={"mapping_team": "$AT"},
    pipe_expr={ "$eq": [ "$team_train",  "$$mapping_team" ] },
    project_clause={"tfmkt": 1, "_id": 0},
    set_clause={"AT_tfmkt": "$AT_mapping.tfmkt"}
)

map_fbd_game = [
    {"$lookup": {
        "from": "footballdata_matches",
        "let": {
            "mapping_home_team": "$HT_fbd", "mapping_away_team": "$AT_fbd",
            "mapping_date": "$iso_date"
        },
        "pipeline": [
            {"$match": { 
                "$expr": { 
                    "$and": [
                         { "$eq": [ "$HomeTeam",  "$$mapping_home_team" ] },
                         { "$eq": [ "$AwayTeam",  "$$mapping_away_team" ] },
                         { "$eq": [ "$iso_date",  "$$mapping_date" ] }
                    ]
                }
            }},
        ],
        "as": "fbd_mapped_matches"
    }}
]

map_tfmkt_game = [
    {"$lookup": {
        "from": "tfmkt_games",
        "let": {
            "mapping_home_team": "$HT_tfmkt", "mapping_away_team": "$AT_tfmkt",
            "mapping_date": "$iso_date"
        },
        "pipeline": [
            {"$match": { 
                "$expr": { 
                    "$and": [
                         { "$eq": [ "$home_team",  "$$mapping_home_team" ] },
                         { "$eq": [ "$away_team",  "$$mapping_away_team" ] },
                         { "$eq": [ "$iso_date",  "$$mapping_date" ] }
                    ]
                }
            }},
        ],
        "as": "tfmkt_mapped_matches"
    }}
]

query_pipeline = map_HT_fbd + map_AT_fbd + map_HT_tfmkt + map_AT_tfmkt + map_tfmkt_game
training_collection = mongo_connection.get_database('soccer-challenge').get_collection('training_matches')
results = training_collection.aggregate(query_pipeline)

report = {}
for r in results:
    training_unique_id = r['unique_id']
    for mapping_collection, external_id in [
        #('fbd_mapped_matches', "Serial"), 
        ('tfmkt_mapped_matches', 'game_id')
    ]:
        mapped_matches = r.get(mapping_collection, [])
        if len(mapped_matches) == 0:
            external_id = "MISSING"
        elif len(mapped_matches) > 1:
            external_id = "AMBIGUOUS"
        else:
            match_object = mapped_matches[0]
            match_object['unique_id'] = training_unique_id
            external_id = match_object[external_id]
        report.setdefault(mapping_collection, []).append({
            "unique_id": training_unique_id,
            "season": r['Sea'],
            "league": r["Lge"],
            "date": r['iso_date'],
            "home_team": r['HT'],
            "away_team": r['AT'],
            "external_id":  external_id
        }) 

import pandas as pd

tfmkt_df = pd.DataFrame(report["tfmkt_mapped_matches"])
len(tfmkt_df[tfmkt_df['external_id'] == "MISSING"]) / len(tfmkt_df)
len(tfmkt_df[tfmkt_df['external_id'] == "AMBIGUOUS"]) / len(tfmkt_df)


fbd_df = pd.DataFrame(report["fbd_mapped_matches"])
len(fbd_df[fbd_df['external_id'] == "MISSING"]) / len(fbd_df)
len(fbd_df[fbd_df['external_id'] == "AMBIGUOUS"]) / len(fbd_df)

for collection, pointer, external_id in [
    #('tfmkt_games', 'tfmkt_mapped_matches', 'game_id'),
    ('footballdata_matches', 'fbd_mapped_matches', 'Serial')
]:
    collection = mongo_connection.get_database('soccer-challenge').get_collection(collection)
    outcome = collection.bulk_write([
        UpdateOne(
            filter={external_id: item['external_id']},
            update={"$set": {"training_id": item['unique_id']}}
        ) for item in report[pointer]
    ])
    print(outcome.bulk_api_result)
