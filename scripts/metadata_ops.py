import click
import json
import os

import numpy as np
import pandas as pd
from pymongo.operations import ReplaceOne, UpdateMany, UpdateOne
from tqdm import tqdm

from scripts.helper import get_team_map, get_team_map_by_distance
from scripts.metadata import TRAINING_FILTERS, FILTERS
from utils import mongo_db

@click.group()
def cli():
    pass

@cli.command()
def country_codes_hash():
    """We retrieve the distinct country codes currently in the training dataset
    and trying to match them to the ISO standard.

    """
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    training_countries = mongo_connection.get_database('soccer-challenge').get_collection(
        'training_matches'
    ).distinct("Lge")
    tfmkt_countries = mongo_connection.get_database('soccer-challenge').get_collection(
        'tfmkt_competitions'
    ).find()
    tfmkt_dict = {}
    for tf_country in tfmkt_countries:
        if 'country_name' in tf_country:
            tfmkt_dict[(tf_country['country_name'], tf_country['competition_type'])] = tf_country
    click.echo(f"Retrieved a total of {len(training_countries)} countries.")  
    iso_codes_file = open(os.path.join("input", "metadata", "country-codes-all.json"), 'r')
    iso_dict = {}
    for elem in json.load(iso_codes_file):
        iso_dict[elem['alpha-3']] = elem
    country_documents = []
    tier_options = ['first_tier', 'second_tier', 'third_tier', 'fourth_tier', 'fifth_tier']
    for country in training_countries:
        training_country_code = country[:-1]
        training_tier = country[-1]
        tfmkt_tier = tier_options[int(training_tier) - 1]
        iso_country = iso_dict.get(training_country_code, {})
        new_country = {
            "training_code": country,
            "training_country_code": training_country_code,
            "training_tier": training_tier,
            "ISO_alpha_3": iso_country.get('alpha-3', '404'),
            "ISO_alpha_2": iso_country.get('alpha-2', '404'),
            "ISO_name": iso_country.get('name', 'Not found'),
            "ISO_country_code": iso_country.get('country-code', '404')
        }
        new_country["tfmkt_href"] = tfmkt_dict.get((new_country['ISO_name'], tfmkt_tier), {}).get('href', '404')
        country_documents.append(new_country)
    json.dump([cd for cd in country_documents if cd['tfmkt_href'] == "404"], open(os.path.join("output", "scraping_staging", "country-codes-hash.json"), 'w'))
    json.dump([cd for cd in country_documents if cd['tfmkt_href'] != "404"], open(os.path.join("output", "scraping_staging", "country-codes-hash-update.json"), 'w'))

@cli.command()
@click.argument('external_collection', type=str)
@click.argument('team_prefix', type=str)
@click.option('-u', '--update-on-db', 'update_on_db', is_flag=True, show_default=True, default=False, help="Update records on db.")
@click.option('-p', '--print', 'print_as_csv', is_flag=True, show_default=True, default=False, help="Print also as a csv file.")
@click.option('-l', '--leagues', 'leagues', multiple=True)
def map_leagues_team_names(external_collection, team_prefix, update_on_db, print_as_csv, leagues):
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    training_db = mongo_connection.get_database('soccer-challenge').get_collection('training_matches')
    result = training_db.aggregate([
        {"$group": {
            "_id": {"Sea": "$Sea", "Lge": "$Lge"}, 
            "total_count": {"$count": {}}, 
            "filtered": {"$push": {"Date": "$iso_date", "HT": "$HT", "AT": "$AT"}}
        }},
        {"$sort": {"_id": 1}}
    ])
    external_db = mongo_connection.get_database('soccer-challenge').get_collection(external_collection)
    external_result = external_db.aggregate(get_aggregate_result(external_collection))
    training_season_leagues = {(r['_id']['Sea'], r['_id']['Lge']): r['filtered'] for r in result}
    external_season_leagues = {(r['_id']['Sea'], r['_id']['Lge']): r['filtered'] for r in external_result}
    team_mapping = []
    messages = []
    for sea, lge in (pbar := tqdm(training_season_leagues.keys())):
        if (leagues) and (lge not in leagues):
            continue
        pbar.set_description(f"Season {sea} for league {lge}.")
        training_data = pd.DataFrame(training_season_leagues.get((sea, lge), []))
        external_data = pd.DataFrame(external_season_leagues.get((sea, lge), []))
        team_map, converging_status = get_team_map(training_data, external_data, ht_ext = 'HT', at_ext='AT')
        if converging_status in ["Partially", "Fully"]:
            team_mapping += [{"Sea": sea, "Lge": lge, "team_train": k, team_prefix: v} for k, v in team_map.items()]
        messages.append(f"Season {sea} for league {lge} {converging_status}.")
    if update_on_db:
        teams_collection = mongo_connection.get_database('soccer-challenge').get_collection('teams_by_season')
        outcome = teams_collection.bulk_write(
            [
                UpdateOne(
                    filter={"Sea": item['Sea'], "Lge": item["Lge"], "team_train": item['team_train']},
                    update={"$set": {team_prefix: item[team_prefix]}},
                    upsert=True
                ) 
            for item in team_mapping]
        )
        click.echo(f"{outcome.upserted_count + outcome.inserted_count} items inserted.")
        click.echo(f"{outcome.modified_count} items updated.")
    click.echo(f"{len(messages)} errors found:")
    click.echo("\n".join(messages))
    if print_as_csv:
        pd.DataFrame(team_mapping).to_csv('mapping_teams.csv')

@cli.command()
def add_empty_team_mappings():
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    teams_by_season = mongo_connection.get_database('soccer-challenge').get_collection('training_matches')
    grouped_matches = teams_by_season.aggregate([
        {"$group": {
            "_id": {"season": "$Sea", "league": "$Lge"},
            "home_teams": {
                "$addToSet": "$HT"
            },
            "away_teams": {
                "$addToSet": "$AT"
            }
        }}
    ])
    updates = []
    for record in grouped_matches:
        season = record['_id']['season']
        league = record['_id']['league']
        season = record['_id']['season']
        home_teams = set(record['home_teams'])
        away_teams = set(record['away_teams'])
        teams = list(home_teams.union(away_teams))
        for t in teams:
            updates.append(
                UpdateOne(
                    filter={"Sea": season, "Lge": league, "team_train": t},
                    update={"$set": {"team_train": t}},
                    upsert=True
                )
            )
    click.echo(f"Dry run {len(updates)} updates.")
    teams_by_season = mongo_connection.get_database('soccer-challenge').get_collection('teams_by_season')
    outcome = teams_by_season.bulk_write(updates)
    click.echo(f"{outcome.upserted_count + outcome.inserted_count} items inserted.")
    click.echo(f"{outcome.modified_count} items updated.")

@cli.command()
@click.argument('external_collection', type=str)
@click.option('-l', '--leagues', 'leagues', multiple=True)
@click.option('-u', '--update-on-db', 'update_on_db', is_flag=True, show_default=True, default=False, help="Update records on db.")
def adding_tfmkt_team_names(external_collection, leagues, update_on_db):
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    match_pipeline = [{
        "$match": {"Lge": {"$in": leagues}}
    }] if leagues else []
    training_db = mongo_connection.get_database('soccer-challenge').get_collection('training_matches')
    result = training_db.aggregate(match_pipeline + [
        {"$group": {
            "_id": {"Lge": "$Lge"}, 
            "total_count": {"$count": {}}, 
            "filtered": {"$push": {"Date": "$iso_date", "HT": "$HT", "AT": "$AT"}}
        }},
        {"$sort": {"_id": 1}}
    ])
    external_db = mongo_connection.get_database('soccer-challenge').get_collection(external_collection)
    external_result = external_db.aggregate([
        {"$match": {"competition": {"$ne": ''}}},
        {"$group": {
            "_id": {"Lge": "$competition"}, 
            "total_count": {"$count": {}}, 
            "filtered": {"$push": {"Date": "$iso_date", "HT": "$home_team", "AT": "$away_team"}}
        }},
        {"$sort": {"_id": 1}}
    ])
    competitions_mapping = mongo_connection.get_database('soccer-challenge').get_collection('countries')
    comp_results = competitions_mapping.find(filter={}, projection={"tfmkt_href": 1, "training_code": 1, "_id": 0})
    training_compt_to_tfmkt = {db_record['training_code']: db_record['tfmkt_href'] for db_record in comp_results}
    training_season_leagues = {r['_id']['Lge']: (r['total_count'], r['filtered']) for r in result}
    external_season_leagues = {r['_id']['Lge']: (r['total_count'], r['filtered']) for r in external_result}
    messages = []
    bulk_updates = []
    for lge in training_season_leagues.keys():
        print(training_compt_to_tfmkt.get(lge, "-"))
        total_count, training_rows = training_season_leagues.get(lge)
        total_ext_count, external_rows = external_season_leagues.get(training_compt_to_tfmkt.get(lge, "-"), (0, []))
        click.echo(f"League {lge} with {total_count} number of matches ({len(external_rows)}) external")
        training_data = pd.DataFrame(training_rows)
        external_data = pd.DataFrame(external_rows)
        team_map, converging_status = get_team_map(training_data, external_data, ht_ext = 'HT', at_ext='AT')
        if converging_status in ["Partially", "Fully"]:
            if update_on_db:
                bulk_updates += [
                    UpdateMany(
                        filter={"team_train": train_team},
                        update={
                            "$set": {"team_tfmkt": tfmkt_team}
                        }
                    )
                for train_team, tfmkt_team in team_map.items()]    
        else:
            messages.append(f"League {lge} {converging_status}.")
    print(messages)
    if update_on_db:
        click.echo(f"Updating {len(bulk_updates)} teams")
        teams_collection = mongo_connection.get_database('soccer-challenge').get_collection('teams_by_season')
        outcome = teams_collection.bulk_write(bulk_updates)
        click.echo(f"{outcome.modified_count} items updated.")
    else:
        click.echo(f"Result {converging_status}")
        click.echo(team_map)
        train_teams = [t for t in training_data['HT'].unique() if t not in team_map]
        external_teams = [t for t in external_data['HT'].unique() if t not in team_map.values()]
        click.echo("Training teams:")
        click.echo(train_teams)
        click.echo("External teams:")
        click.echo(external_teams)
        new_team_map = get_team_map_by_distance(train_teams, external_teams)
        click.echo("New team map:")
        click.echo(new_team_map)

@cli.command()
@click.argument('mappings_file', type=click.Path(exists=True))
def add_extra_mappings(mappings_file):
    extra_mappings = json.load(open(mappings_file, 'r'))
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    bulk_updates = [
        UpdateMany(
            filter={"team_train": train_team},
            update={
                "$set": {"team_tfmkt": tfmkt_team}
            }
        )
        for train_team, tfmkt_team in extra_mappings.items()
    ]
    click.echo(f"Updating {len(bulk_updates)} teams")
    teams_collection = mongo_connection.get_database('soccer-challenge').get_collection('teams_by_season')
    outcome = teams_collection.bulk_write(bulk_updates)
    click.echo(f"{outcome.modified_count} items updated.")

@cli.command()
@click.option('-u', '--update-on-db', 'update_on_db', is_flag=True, show_default=True, default=False, help="Update records on db.")
def unify_team_mappings(update_on_db):
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    teams_by_season = mongo_connection.get_database('soccer-challenge').get_collection('teams_by_season')
    grouped_mappings = teams_by_season.aggregate([
        {"$group": {
            "_id": {"team_train": "$team_train"},
            "season_mappings": {
                "$push": {"season": "$Sea", "fbd": "$team_fbd", "tfmkt": "$team_tfmkt"}
            }
        }}
    ])
    name_changes = []
    name_inserts = []
    for team_mapping in grouped_mappings:
        team_train = team_mapping['_id']['team_train']
        season_mappings = team_mapping['season_mappings']
        unified_mapping = {"team_train": team_train}
        for team_translation in ['tfmkt']:
            team_names = [t[team_translation] for t in season_mappings if team_translation in t]
            if len(team_names) > 0:
                if np.all(np.array(team_names) == team_names[0]):
                    unified_mapping[team_translation] = team_names[0]
                    name_inserts.append(unified_mapping)
                else:
                    name_changes.append(team_train)
    print(f"Incorrect teams: {name_changes}")
    if update_on_db:
        teams_collection = mongo_connection.get_database('soccer-challenge').get_collection('teams')
        outcome = teams_collection.insert_many(documents=name_inserts)
        print(outcome.acknowledged)

@cli.command()
@click.argument('external_collection')
@click.argument('source_collection', type=str, default="training_matches")
@click.option('-u', '--update-on-db', 'update_on_db', is_flag=True, show_default=True, default=False, help="Update records on db.")
def add_fk(external_collection, source_collection, update_on_db):
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    map_HT, map_AT, map_game, external_pointer = get_meta_map(external_collection)
    training_collection = mongo_connection.get_database('soccer-challenge').get_collection(source_collection)
    results = training_collection.aggregate(map_HT + map_AT + map_game)
    report = []
    click.echo("Iterating over the results")
    cursor_i = 1
    for r in results:
        print(cursor_i, end='\r')
        cursor_i += 1
        training_unique_id = r['unique_id']
        mapped_matches = r.get('mapped_matches', [])
        if len(mapped_matches) == 0:
            external_id = "MISSING"
        elif len(mapped_matches) > 1:
            external_id = "AMBIGUOUS"
        else:
            match_object = mapped_matches[0]
            match_object['unique_id'] = training_unique_id
            external_id = match_object[external_pointer]
        report.append({
            "unique_id": training_unique_id,
            "season": r['Sea'],
            "league": r["Lge"],
            "date": r['iso_date'],
            "home_team": r['HT'],
            "away_team": r['AT'],
            "external_id":  external_id
        })
    report_df = pd.DataFrame(report)
    report_df.to_csv(f"map_{external_collection}.csv")
    click.echo("Report generated")
    missing_length = len(report_df[report_df['external_id'] == 'MISSING'])
    ambiguous_length = len(report_df[report_df['external_id'] == 'AMBIGUOUS'])
    click.echo(f"{missing_length/len(report_df) * 100:.2f} % of missing matches.")
    click.echo(f"{ambiguous_length/len(report_df) * 100:.2f} % of duplicate matches.")
    if update_on_db:
        collection = mongo_connection.get_database('soccer-challenge').get_collection(external_collection)
        outcome = collection.bulk_write([
            UpdateOne(
                filter={external_pointer: item['external_id']},
                update={"$set": {"training_id": item['unique_id']}}
            ) for item in report if item['external_id'] not in ['MISSING', 'AMBIGUOUS']
        ])
        click.echo(outcome.bulk_api_result)
    else:
        click.echo(f"Dry run {len([r for r in report if r['external_id'] not in ['MISSING', 'AMBIGUOUS']])}")

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

def get_meta_map(source_id):
    if source_id == 'tfmkt_games':
        map_HT = get_map(
            prefix="HT",
            from_collection="teams",
            let_clause={"mapping_team": "$HT"},
            pipe_expr={ "$eq": [ "$team_train",  "$$mapping_team" ] },
            project_clause={"tfmkt": 1, "_id": 0},
            set_clause={"HT_tfmkt": "$HT_mapping.tfmkt"}
        )
        map_AT = get_map(
            prefix="AT",
            from_collection="teams",
            let_clause={"mapping_team": "$AT"},
            pipe_expr={ "$eq": [ "$team_train",  "$$mapping_team" ] },
            project_clause={"tfmkt": 1, "_id": 0},
            set_clause={"AT_tfmkt": "$AT_mapping.tfmkt"}
        )
        map_game = [
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
                "as": "mapped_matches"
            }}
        ]
        external_pointer = "game_id"
    elif source_id == 'footballdata_matches':
        map_HT = get_map(
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

        map_AT = get_map(
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
        map_game = [
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
                "as": "mapped_matches"
            }}
        ]
        external_pointer = 'Serial'
    elif source_id == 'spi_matches':
        map_HT = get_map(
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
            project_clause={"team_spi": 1, "_id": 0},
            set_clause={"HT_spi": "$HT_mapping.team_spi"}
        )

        map_AT = get_map(
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
            project_clause={"team_spi": 1, "_id": 0},
            set_clause={"AT_spi": "$AT_mapping.team_spi"}
        )
        map_game = [
            {"$lookup": {
                "from": "spi_matches",
                "let": {
                    "mapping_home_team": "$HT_spi", "mapping_away_team": "$AT_spi",
                    "mapping_date": "$iso_date"
                },
                "pipeline": [
                    {"$match": { 
                        "$expr": { 
                            "$and": [
                                { "$eq": [ "$team1",  "$$mapping_home_team" ] },
                                { "$eq": [ "$team2",  "$$mapping_away_team" ] },
                                { "$eq": [ "$iso_date",  "$$mapping_date" ] }
                            ]
                        }
                    }},
                ],
                "as": "mapped_matches"
            }}
        ]
        external_pointer = 'Serial'
    elif source_id == 'odds_portal_matches':
        map_HT = get_map(
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
            project_clause={"team_op": 1, "_id": 0},
            set_clause={"HT_op": "$HT_mapping.team_op"}
        )

        map_AT = get_map(
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
            project_clause={"team_op": 1, "_id": 0},
            set_clause={"AT_op": "$AT_mapping.team_op"}
        )
        map_game = [
            {"$lookup": {
                "from": "odds_portal_matches",
                "let": {
                    "mapping_home_team": "$HT_op", "mapping_away_team": "$AT_op",
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
                "as": "mapped_matches"
            }}
        ]
        external_pointer = 'Serial'
    elif source_id == 'odds_portal_next_matches':
        map_HT = get_map(
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
            project_clause={"team_op": 1, "_id": 0},
            set_clause={"HT_op": "$HT_mapping.team_op"}
        )

        map_AT = get_map(
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
            project_clause={"team_op": 1, "_id": 0},
            set_clause={"AT_op": "$AT_mapping.team_op"}
        )
        map_game = [
            {"$lookup": {
                "from": "odds_portal_next_matches",
                "let": {
                    "mapping_home_team": "$HT_op", "mapping_away_team": "$AT_op",
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
                "as": "mapped_matches"
            }}
        ]
        external_pointer = 'Serial'
    return map_HT, map_AT, map_game, external_pointer

def get_aggregate_result(source_id):
    aggregation_pipeline = []
    if source_id == "footballdata_matches":
        aggregation_pipeline = [
            {"$match": {"Sea": {"$ne": None}, "comp_div": {"$ne": ''}}},
            {"$group": {
                "_id": {"Sea": "$Sea", "Lge": "$comp_div"}, 
                "total_count": {"$count": {}}, 
                "filtered": {"$push": {"fb_div": "$Div", "Date": "$iso_date", "HT": "$HomeTeam", "AT": "$AwayTeam"}}
            }},
            {"$sort": {"_id": 1}}
        ]
    elif source_id == 'spi_matches':
        aggregation_pipeline = [
            {"$group": {
                "_id": {"Sea": "$Sea", "Lge": "$Lge"}, 
                "total_count": {"$count": {}}, 
                "filtered": {"$push": {"spi_div": "$league", "Date": "$iso_date", "HT": "$team1", "AT": "$team2"}}
            }},
            {"$sort": {"_id": 1}}
        ]
    elif source_id == 'odds_portal_matches':
        aggregation_pipeline = [
            {"$group": {
                "_id": {"Sea": "$Sea", "Lge": "$Lge"}, 
                "total_count": {"$count": {}}, 
                "filtered": {"$push": {"Date": "$iso_date", "HT": "$home_team", "AT": "$away_team"}}
            }},
            {"$sort": {"_id": 1}}
        ]
    return aggregation_pipeline

@cli.command()
@click.argument('new_collection')
@click.option('-u', '--update-on-db', 'update_on_db', is_flag=True, show_default=True, default=False, help="Update records on db.")
def process_tfmkt_games(new_collection, update_on_db):
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    match_clause = {
        "training_id": {"$exists": True},
    }
    tfmkt_games = mongo_connection.get_database('soccer-challenge').get_collection('tfmkt_games')
    results = tfmkt_games.aggregate([
        {"$match": match_clause},
        {"$lookup": {
            "from": "tfmkt_events",
            'let': {'gameId': "$game_id"},
            'pipeline': [
                {'$match': {'$expr': { '$eq': ['$game_id', '$$gameId'] } }},
                {'$sort': {'event_num': 1}}
            ],
            "as": "events"
        }},
        {"$lookup": {
            "from": "tfmkt_lineups",
            "localField": "game_id",
            "foreignField": 'game_id',
            "as": "lineup"
        }}
    ])

    game_metrics_documents = []
    for match_object in results:
        game_metrics_documents.append(process_match(match_object))
    click.echo(f"New metrics for  {len(game_metrics_documents)} matches")
    if update_on_db:
        metrics_collection = mongo_connection.get_database('soccer-challenge').get_collection(new_collection)
        outcome = metrics_collection.insert_many(documents=game_metrics_documents)
        print(outcome.acknowledged)

def process_match(match_object):
    home_team = match_object.get('home_team', '-')
    away_team = match_object.get('away_team', '-')
    new_metrics = {
        "result": match_object.get('result'), 
        "training_id": match_object.get('training_id'),
        "home_score": 0.0,
        "away_score": 0.0
    }
    for event in match_object.get('events', []):
        event_type = event.get('type', '-')
        if event_type in ['Goals']:
            team = 'home_score' if home_team == event.get('club', '-') else 'away_score' if away_team == event.get('club', '-') else "invalid_score"
            score_increment = 1.0
            event_minute = event.get('minute', 1)
            if event_minute > 70:
                other_team = 'home_score' if team == 'away_score' else 'away_score'
                score_diff = new_metrics.get(team, 0.0) - new_metrics.get(other_team, 0.0)
                if score_diff > 1:
                    minute_diff = max(0, event_minute - 70)
                    score_increment -= 0.025 * minute_diff
            new_metrics[team] = new_metrics.get(team, 0.0) + score_increment
    for lineup_info in match_object.get('lineup', []):
        lineup_team = 'home' if home_team == lineup_info.get('team', '-') else 'away' if away_team == lineup_info.get('team', '-') else "invalid"
        lineup_type = lineup_info.get('lineup_type', 'invalid')
        lineup_key = f"{lineup_team}_{lineup_type}"
        total_key = f"{lineup_key}_total"
        new_metrics[f"{lineup_key}_goakeeper"] = sum(lineup_info.get('Goalkeeper', [0]))
        new_metrics[f"{lineup_key}_defense"] =sum([sum(lineup_info.get(position, [0])) for position in ['Centre-Back', 'Left-Back', 'Right-Back']])
        new_metrics[f"{lineup_key}_midfield"] =sum([sum(lineup_info.get(position, [0])) for position in ['DefensiveMidfield', 'CentralMidfield', 'AttackingMidfield']])
        new_metrics[f"{lineup_key}_wingers"] =sum([sum(lineup_info.get(position, [0])) for position in ['RightWinger', 'LeftWinger']])
        new_metrics[f"{lineup_key}_forward"] =sum([sum(lineup_info.get(position, [0])) for position in ['Centre-Forward']])
        new_metrics[total_key] = new_metrics[f"{lineup_key}_goakeeper"] + new_metrics[f"{lineup_key}_defense"] + new_metrics[f"{lineup_key}_midfield"] + new_metrics[f"{lineup_key}_wingers"] + new_metrics[f"{lineup_key}_forward"]

    return new_metrics


if __name__ == '__main__':
    cli()