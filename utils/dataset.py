import pandas as pd

from typing import Dict, List

TRAINING_COLUMNS = [
    "Sea",
    "Lge",
    "iso_date",
    "HT",
    "AT",
    "HS",
    "AS",
    "GD",
    "WDL",
    "unique_id",
]
PREDICTION_COLUMNS = [
    "Sea",
    "Lge",
    "iso_date",
    "HT",
    "AT",
    "HS",
    "AS",
    "GD",
    "WDL",
    "prd_W",
    "prd_D",
    "prd_L",
    "unique_id",
]

SOURCES = {
    "tfmkt": {
        "collection": "tfmkt_games",
        "columns": [
            "home_club_position",
            "away_club_position",
            "match_day",
            "stadium",
            "home_club_formation",
            "away_club_formation",
        ],
    },
    "fbd": {
        "collection": "footballdata_matches",
        "columns": [
            "avg_home_odds",
            "avg_draw_odds",
            "avg_away_odds",
            "HTS",
            "ATS",
            "HST",
            "AST",
            "HF",
            "AF",
            "HY",
            "AY",
            "HR",
            "AR",
        ],
    },
    "tfmkt_metrics": {
        "collection": "tfmkt_metrics",
        "columns": ["home_starter_total", "away_starter_total"],
    },
    "spi": {
        "collection": "spi_matches",
        "columns": [
            "spi1",
            "spi2",
            "prob1",
            "prob2",
            "probtie",
            "proj_score1",
            "proj_score2",
            "importance1",
            "importance2",
            "score1",
            "score2",
            "xg1",
            "xg2",
            "nsxg1",
            "nsxg2",
            "adj_score1",
            "adj_score2",
        ],
    },
    "odds_portal": {
        "collection": "odds_portal_matches",
        "columns": ["odds_home", "odds_draw", "odds_away"],
    },
    "odds_portal_next": {
        "collection": "odds_portal_next_matches",
        "columns": ["odds_home", "odds_draw", "odds_away"],
    },
}


def get_aggregate_pipeline(
    training_filter: Dict = {},
    extra_sources: List[str] = ["tfmkt", "fbd"],
    columns: List[str] = [],
    excluded_columns: List[str] = [],
) -> List[dict]:
    """
    Returns a list of MongoDB aggregation pipeline stages to extract and transform
    data from the 'training_matches' collection based on the specified parameters.

    Parameters:
    -----------
    training_filter : dict, optional (default={})
        A dictionary object representing the filter criteria to match documents
        in the 'training_matches' collection.
    extra_sources : list of str, optional (default=['tfmkt', 'fbd'])
        A list of strings representing the additional data sources to join with the
        'training_matches' collection. Each source should have a corresponding key
        in the 'SOURCES' dictionary.
    columns : list of str, optional (default=[])
        A list of strings representing the column names to include in the output.
    excluded_columns : list of str, optional (default=[])
        A list of strings representing the column names to exclude from the output.

    Returns:
    --------
    pipeline : list of dict
        List of dictionary objects representing the aggregation pipeline stages.

    """
    # Define the MongoDB aggregation pipeline stages to extract and transform data from the collection
    lookups = [
        {
            "$lookup": {
                "from": SOURCES[s]["collection"],
                "localField": "unique_id",
                "foreignField": "training_id",
                "as": s,
            }
        }
        for s in extra_sources
    ] + [{"$project": {f"{s}._id": 0}} for s in extra_sources]
    unwinds = [
        {"$unwind": {f"path": f"${source}", "preserveNullAndEmptyArrays": True}}
        for source in extra_sources
    ]
    if "tfmkt" in extra_sources:
        lookups += [
            {
                "$lookup": {
                    "from": "tfmkt_lineups",
                    "localField": "tfmkt.game_id",
                    "foreignField": "game_id",
                    "as": "lineup_info",
                }
            },
            {
                "$lookup": {
                    "from": "tfmkt_events",
                    "localField": "tfmkt.game_id",
                    "foreignField": "game_id",
                    "as": "events_info",
                }
            },
            {"$project": {"events_info._id": 0, "lineup_info._id": 0}},
        ]
    default_project = {"_id": 0}
    for c in columns:
        default_project[c] = 1
    for c in excluded_columns:
        default_project[c] = 0
    # Construct the pipeline stages
    pipeline = (
        [{"$match": training_filter}]
        + lookups
        + unwinds
        + [{"$project": default_project}]
    )
    # Return the pipeline stages
    return pipeline


def get_flatten_pipeline(source: str) -> List[dict]:
    """
    Returns a list of MongoDB aggregation pipeline stages to flatten a nested document
    from the specified source.

    Parameters:
    -----------
    source : str
        The name of the nested document to flatten.

    Returns:
    --------
    pipeline : list of dict
        List of dictionary objects representing the aggregation pipeline stages.

    """
    # Define the aggregation pipeline stages to flatten the specified source document
    pipeline = [
        {"$project": {"res": {"$mergeObjects": ["$$ROOT", f"${source}"]}}},
        {"$replaceRoot": {"newRoot": "$res"}},
        {"$unset": source},
    ]
    # Return the pipeline stages
    return pipeline


def get_database_collections(mongo_conn):
    collections = mongo_conn.get_database("soccer-challenge").list_collection_names()
    return collections


def get_raw_data(
    mongo_conn,
    training_collection="training_matches",
    training_filter: Dict = {},
    extra_sources: List[str] = ["tfmkt", "fbd"],
    columns: List[str] = [],
    excluded_columns: List[str] = [],
    flatten: bool = False,
) -> List[Dict]:
    """
    Retrieves raw training data from the database and returns it as a list of dictionaries.

    Parameters:
    -----------
    mongo_conn : MongoConnection
        Connection to the database from which to retrieve the data.
    training_filter : dict, default={}
        Dictionary containing filters to apply to the query. Filters should be specified
        in the form {'column_name': value}. If no filters are provided, returns all rows.
    extra_sources : list of str, default=['tfmkt', 'fbd']
        List of additional sources to retrieve data from.
    columns : list of str, default=[]
        List of columns to retrieve from the database.
    excluded_columns : list of str, default=[]
        List of columns to exclude from the retrieved data.
    flatten : bool, default=False
        Flag indicating whether to flatten nested structures in the retrieved data.

    Returns:
    --------
    cursor_results : list of dict
        List of dictionaries containing the raw training data retrieved from the database.

    """
    # Get the aggregate pipeline to retrieve the data
    agg_pipeline = get_aggregate_pipeline(
        training_filter, extra_sources, columns, excluded_columns
    )
    # Flatten any nested structures in the retrieved data
    if flatten:
        for s in extra_sources:
            agg_pipeline += get_flatten_pipeline(s)
    # Retrieve data from the database using the aggregate function
    cursor_results = (
        mongo_conn.get_database("soccer-challenge")
        .get_collection(training_collection)
        .aggregate(agg_pipeline)
    )
    # Convert the cursor to a list of dictionaries
    cursor_results = list(cursor_results)
    # Return the list of dictionaries
    return cursor_results


def get_raw_data_as_df(
    mongo_conn,
    training_filter: Dict = {},
) -> pd.DataFrame:
    """
    Retrieves raw training data from the database and returns it as a pandas DataFrame.

    Parameters:
    -----------
    db_conn : MongoConnection
        Connection to the database from which to retrieve the data.
    training_filter : dict, default={}
        Dictionary containing filters to apply to the query. Filters should be specified
        in the form {'column_name': value}. If no filters are provided, returns all rows.

    Returns:
    --------
    df : pandas DataFrame
        DataFrame containing the raw training data retrieved from the database.

    """
    # Define columns to retrieve from the database
    columns = (
        TRAINING_COLUMNS
        + [f"fbd.{c}" for c in SOURCES["fbd"]["columns"]]
        + [f"tfmkt_metrics.{c}" for c in SOURCES["tfmkt_metrics"]["columns"]]
    )
    # Add extra sources to retrieve data from
    extra_sources = ["fbd", "tfmkt_metrics"]
    # Retrieve data from the database using the get_raw_data function
    cursor_results = get_raw_data(
        mongo_conn=mongo_conn,
        training_filter=training_filter,
        extra_sources=extra_sources,
        columns=columns,
        flatten=True,
    )
    # Convert cursor_results to a pandas DataFrame
    df = pd.DataFrame(list(cursor_results))
    # Return the DataFrame
    return df


def get_prediction_set_as_df(
    mongo_conn,
    training_filter: Dict = {},
) -> pd.DataFrame:
    """
    Retrieves raw training data from the database and returns it as a pandas DataFrame.

    Parameters:
    -----------
    db_conn : MongoConnection
        Connection to the database from which to retrieve the data.
    training_filter : dict, default={}
        Dictionary containing filters to apply to the query. Filters should be specified
        in the form {'column_name': value}. If no filters are provided, returns all rows.

    Returns:
    --------
    df : pandas DataFrame
        DataFrame containing the raw training data retrieved from the database.

    """
    # Define columns to retrieve from the database
    columns = PREDICTION_COLUMNS + [
        f"odds_portal_next.{c}" for c in SOURCES["odds_portal_next"]["columns"]
    ]
    # Add extra sources to retrieve data from
    extra_sources = ["odds_portal_next"]
    # Retrieve data from the database using the get_raw_data function
    cursor_results = get_raw_data(
        mongo_conn,
        "prediction_matches",
        training_filter,
        extra_sources=extra_sources,
        columns=columns,
        flatten=True,
    )
    # Convert cursor_results to a pandas DataFrame
    df = pd.DataFrame(list(cursor_results))
    # Return the DataFrame
    return df


def get_pretty_df(key_type_count, total_docs):
    """
    Returns DataFrame object built using the key_type dictionary
    :param key_type_count: The distribution of key types
    :type key_type_count: dictionary
    :return: Dataframe built from the key type dict
    """
    table_headers = [
        "Key",
        "Occurrence Count",
        "Occurrence Percentage",
        "Value Type",
        "Value Type Percentage",
    ]
    result_table = []

    for key, key_types in key_type_count.items():
        total_keys = sum(key_types.values())
        max_key_type_count = max(key_types.values())

        max_key_type = [
            key_type
            for key_type, key_type_count in key_types.items()
            if key_type_count == max_key_type_count
        ][0]

        max_key_percent = (
            round(max_key_type_count * 100.0 / total_keys, 2) if total_keys else 0.0
        )
        occurrence_percent = (
            round(total_keys * 100.0 / total_docs, 2) if total_docs else 0.0
        )

        table_row = [key, total_keys, occurrence_percent, max_key_type, max_key_percent]
        result_table.append(table_row)

    return pd.DataFrame(result_table, columns=table_headers)


def get_schema(mongo_conn, collection_name, limit=100, filter={}):
    total_docs = 0
    key_type_default_count = {
        int: 0,
        float: 0,
        str: 0,
        bool: 0,
        dict: 0,
        list: 0,
        set: 0,
        tuple: 0,
        None: 0,
        object: 0,
        "unicode": 0,
        "other": 0,
    }

    mongo_collection_docs = (
        mongo_conn.get_database("soccer-challenge")
        .get_collection(collection_name)
        .find(filter, projection={"_id": 0})
        .limit(limit)
    )

    key_type_count = {}

    for doc in mongo_collection_docs:
        for key, value in doc.items():
            type_of_value = (
                type(value) if type(value) in key_type_default_count else "other"
            )
            if type_of_value == dict:
                for sub_key, sub_value in value.items():
                    sub_type = (
                        type(sub_value)
                        if type(sub_value) in key_type_default_count
                        else "other"
                    )
                    key_type_count.setdefault(f"{key}.{sub_key}", {})[sub_type] = (
                        key_type_count.setdefault(f"{key}.{sub_key}", {}).get(
                            sub_type, 0
                        )
                        + 1
                    )
            elif (type_of_value == list) and (len(value) > 0):
                type_elem = type(value[0])
                key_type_count.setdefault(key, {})[f"{type_of_value}[{type_elem}]"] = (
                    key_type_count.setdefault(key, {}).get(
                        f"{type_of_value}[{type_elem}]", 0
                    )
                    + 1
                )
            else:
                key_type_count.setdefault(key, {})[type_of_value] = (
                    key_type_count.setdefault(key, {}).get(type_of_value, 0) + 1
                )
        total_docs += 1

    result_df = get_pretty_df(key_type_count, total_docs)
    return result_df
