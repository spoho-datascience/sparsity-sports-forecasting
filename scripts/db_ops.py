import click
import csv
import json
import os

import pandas as pd
import numpy as np
from pymongo.operations import ReplaceOne, UpdateOne
from decimal import Decimal
from datetime import datetime
from dateutil import parser
from tqdm import tqdm

from metadata import SEASON_MAPPING, FBDATA_2021_LEAGUES, FBDATA, SPIDATA
from scripts.scrapping_ops import TOURNAMENTS_OP
from utils import mongo_db

TENS = dict(
    k=10e3,
    m=10e6,
    b=10e9
)

@click.group()
def cli():
    pass

@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.argument('target_collection', type=str, default='training_matches')
@click.option('--start', '-s', 'unique_id_start', show_default=True, default=0, type=int)
def upload_training_data(path, target_collection, unique_id_start):
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    training_file = open(path, 'r', encoding='UTF-8-SIG')
    reader = csv.DictReader(training_file)
    matches_list = []
    nr = unique_id_start
    for row in reader:
        if row['Date'] != '':
            row['unique_id'] = nr
            row['iso_date'] = parser.parse(row['Date'], dayfirst=True)
            matches_list.append(row)
            nr += 1
    click.echo(f"Uploading {len(matches_list)} matches.")
    matches_collection = mongo_connection.get_database('soccer-challenge').get_collection(target_collection)
    ops_result = matches_collection.insert_many(
        documents=matches_list
    )
    click.echo(ops_result.acknowledged)

@cli.command()
def remove_training_data():
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    training_matches_collection = mongo_connection.get_database('soccer-challenge').get_collection('training_matches')
    ops_result = training_matches_collection.delete_many(filter={})
    click.echo(ops_result.acknowledged)

@cli.command()
@click.argument("collections", type=str, nargs=-1)
def remove_collections(collections):
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    for c in collections:
        del_col = mongo_connection.get_database('soccer-challenge').get_collection(c)
        ops_result = del_col.drop()

@cli.command()
@click.argument('collection', type=str)
@click.argument('path', type=click.Path(exists=True))
def upload_transfermarkt_staging_data(collection, path):
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    file = open(path, 'r')
    objects_in_file = file.readlines()
    documents = []
    for object in objects_in_file:
        documents.append(json.loads(object))
    click.echo(f"Uploading {len(documents)} to collection {collection}.")
    col = mongo_connection.get_database('soccer-challenge').get_collection(collection)
    ops_result = col.insert_many(documents=documents)
    click.echo(ops_result.acknowledged)

@cli.command()
@click.argument('collection', type=str)
@click.argument('path', type=click.Path(exists=True))
@click.option('--print', '-p', 'print_as_csv', is_flag=True, show_default=True, default=False, help="Print also as a csv file.")
def upload_fbdata(collection, path, print_as_csv):
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    matches_file = open(path, 'r')
    file_size = len(matches_file.readlines())
    matches_file = open(path, 'r')
    reader = csv.DictReader(matches_file)
    matches_list = []
    click.echo("Reading and cleaning data")
    for row in tqdm(reader, total=file_size):
        if row['Date'] != '':
            row['iso_date'] = parser.parse(row['Date'], dayfirst=True)
            row['Sea'] = SEASON_MAPPING.get("2021/2022") if (
                row['comp_div'] in FBDATA_2021_LEAGUES
            ) and (row['Season'] == "2021") else SEASON_MAPPING.get(row['Season'])
            convert_to_numeric(
                row,
                excluded_keys=[
                    "Serial", "Div", "Date", "Time", "FTR", "HTR", "Res", "Country", "Season", "HomeTeam", 
                    "AwayTeam", "Referee", "comp_cc", "comp_desc", "comp_div", "iso_date", "Sea",
                    "BbAHh"
                ]
            )
            add_aggregations(row)
            matches_list.append(row)
    click.echo(f"Uploading {len(matches_list)} matches.")
    fbdata_collection = mongo_connection.get_database('soccer-challenge').get_collection(collection)
    ops_result = fbdata_collection.insert_many(documents=matches_list)
    if ops_result.acknowledged:
        click.echo("Import correct.")
        if print_as_csv:
            pd.DataFrame(matches_list).to_csv('fb_data_imported.csv')
    else:
        click.echo("Something went wrong with MongoDB")

@cli.command()
@click.argument('collection', type=str)
@click.argument('path', type=click.Path(exists=True))
@click.option('--print', '-p', 'print_as_csv', is_flag=True, show_default=True, default=False, help="Print also as a csv file.")
def upload_spi_data(collection, path, print_as_csv):
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    matches_file = open(path, 'r')
    file_size = len(matches_file.readlines())
    matches_file = open(path, 'r')
    reader = csv.DictReader(matches_file)
    matches_list = []
    click.echo("Reading and cleaning data")
    row_i = 0
    for row in tqdm(reader, total=file_size):
        row_i += 1
        row['Serial'] = row_i
        row['iso_date'] = parser.parse(row['date'])
        row['Sea'] = SEASON_MAPPING.get(row['season'])
        row['Lge'] = SPIDATA.get(int(row['league_id']))
        convert_to_numeric(
            row,
            excluded_keys=[
                "season", "date", "league_id", "league", "team1", "team2", "Sea", "Lge", "iso_date"
            ]
        )
        matches_list.append(row)
    click.echo(f"Uploading {len(matches_list)} matches.")
    spi_collection = mongo_connection.get_database('soccer-challenge').get_collection(collection)
    ops_result = spi_collection.insert_many(documents=matches_list)
    if ops_result.acknowledged:
        click.echo("Import correct.")
        if print_as_csv:
            pd.DataFrame(matches_list).to_csv('spi_data_imported.csv')
    else:
        click.echo("Something went wrong with MongoDB")

@cli.command()
@click.argument('collection', type=str)
@click.argument('folder', type=click.Path(exists=True))
@click.option('-l', '--leagues', 'leagues', multiple=True)
@click.option('-p', '--parsed_season', 'parsed_season', type=str, default=None)
@click.option('--update-db', '-u', 'update_db', is_flag=True, show_default=True, default=False, help="Update DB")
@click.option('--upsert', '-x', 'upsert_docs', is_flag=True, show_default=True, default=False, help="Upsert new docs")
def upload_odds_portal(collection, folder, leagues, parsed_season, update_db, upsert_docs):
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    
    list_of_files = os.listdir(folder)
    leagues = leagues or list(TOURNAMENTS_OP.keys())
    for country_code, tournament_info in TOURNAMENTS_OP.items():
        if country_code in leagues:
            files_to_upload = [file_name for file_name in list_of_files if file_name.startswith(country_code)]
            if len(files_to_upload) > 0:
                new_matches = []
                pbar = tqdm(files_to_upload)
                for data_file in pbar:
                    pbar.set_description(f"Processing {data_file}")
                    new_matches += upload_data_file(os.path.join(folder, data_file), country_code=country_code, parsed_season=parsed_season)
                click.echo(f"{country_code}: Uploading {len(new_matches)} matches.")
                if update_db and (len(new_matches) > 0):
                    op_collection = mongo_connection.get_database('soccer-challenge').get_collection(collection)
                    if upsert_docs:
                        outcome = op_collection.bulk_write(
                            [
                                UpdateOne(
                                    filter={"Serial": item['Serial']},
                                    update={"$set": {k: v for k,v in item.items()}},
                                    upsert=True
                                ) 
                            for item in new_matches]
                        )
                        print(outcome.bulk_api_result)
                    else:
                        ops_result = op_collection.insert_many(documents=new_matches)
                        if ops_result.acknowledged:
                            click.echo("Import correct.")

def upload_data_file(file_path, country_code, parsed_season=None):
    file_obj = open(file_path, 'r')
    if not parsed_season:
        season = file_path.replace('-data.csv', '')[-10:]
        split_season = season.split("-")
        if split_season[0] == '':
            parsed_season = f"{split_season[1][-2:]}-{split_season[2][-2:]}"
        else:
            int_season = int(file_path.replace('-data.csv', '')[-2:])
            parsed_season = f"{int_season:02}-{(int_season + 1):02}"
    reader = csv.DictReader(file_obj)
    matches_list = []
    for row in reader:
        if ('match_date' in row) and (row['match_date'] != ''):
            row['Serial'] = f"{country_code}_{parsed_season}_{row.pop('')}"
            row['iso_date'] = datetime.combine(parser.parse(row['match_date'].replace('Today', '').replace('Yesterday', '').replace('Tomorrow', '')).date(), datetime.min.time())
            row['Sea'] = parsed_season
            row['Lge'] = country_code
            row['Original'] = file_path.replace('.csv', '')
            convert_to_numeric(
                row,
                excluded_keys=['Serial', 'Original', 'Sea', 'Lge', 'iso_date', 'home_team', 'away_team', 'match_date']
            )
            matches_list.append(row)
    return matches_list

def convert_to_numeric(match_object, excluded_keys=[]):
    keys_to_numeric = [k for k in match_object.keys() if k not in excluded_keys]
    for k in keys_to_numeric:
        match_object[k] = float(pd.to_numeric(match_object[k], errors='coerce'))

def add_aggregations(match_object):
    home_odds = np.array([match_object[h] for h in FBDATA['HOME_ODDS'] if h in match_object])
    draw_odds = np.array([match_object[d] for d in FBDATA['DRAW_ODDS'] if d in match_object])
    away_odds = np.array([match_object[a] for a in FBDATA['AWAY_ODDS'] if a in match_object])
    match_object['avg_home_odds'] = np.nanmean(home_odds)
    match_object['avg_draw_odds'] = np.nanmean(draw_odds)
    match_object['avg_away_odds'] = np.nanmean(away_odds)
    match_object['available_home_odds'] = np.count_nonzero(~np.isnan(home_odds))
    match_object['available_draw_odds'] = np.count_nonzero(~np.isnan(draw_odds))
    match_object['available_away_odds'] = np.count_nonzero(~np.isnan(away_odds))

@cli.command()
@click.option('--format', type=str, default='scrapy')
def get_list_of_tournaments(format):
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    tournament_list = mongo_connection.get_database("soccer-challenge").get_collection("countries").find()
    tournaments = []
    if format == "scrapy":
        for t in tournament_list:
            tournaments.append({
                "type": "competition",
                "competition_type": "first_tier",
                "href": t['tfmkt_href']

            })
        output_file = open(os.path.join("output", "list_of_tournaments.json"), 'w')
        for tournament_to_print in tournaments:
            print(json.dumps(tournament_to_print), file=output_file)
        
@cli.command()
@click.argument('input_files', type=click.Path(exists=True))
@click.option('--upsert', '-u', 'upsert', is_flag=True, show_default=True, default=True, help="Only add if they don't exist.")
def upload_tfmkt_games_data(input_files, upsert):
    files_to_read = []
    append_path = False
    success, mongo_connection, info = mongo_db.open_mongo_secure_connection()
    if not success:
        click.echo(info.get("msg", ""))
        return
    if os.path.isdir(input_files):
        files_to_read = [filename for filename in os.listdir(input_files)]
        append_path = True
    else:
        files_to_read.append(input_files)
    game_ids_seen = []
    for filename in files_to_read:
        click.echo(f"File: {filename}")
        if append_path:
            file = open(os.path.join(input_files, filename), 'r')
        else:
            file = open(filename, 'r')
        new_games = []
        new_events = []
        new_lineups = []
        object_lines = file.readlines()
        for json_line in tqdm(object_lines):
            new_game = {}
            game_object = json.loads(json_line)
            game_info = game_object.get('main', {})
            new_game = {
                "competition": game_info.get('parent', {}).get('href'),
                "href": game_info.get('href'),
                "game_id": game_info.get('game_id'),
                "home_club_href": game_info.get('home_club', {}).get('href'),
                "away_club_href": game_info.get('away_club', {}).get('href'),
                "home_club_position": int((game_info.get('home_club_position', '0') or "0").replace("Position:", "")),
                "away_club_position": int((game_info.get('away_club_position', '0') or "0").replace("Position:", "")),
                "result": game_info.get('result'),
                "match_day": game_info.get('matchday'),
                "date": game_info.get('date'),
                "stadium": game_info.get('stadium'),
                "referee": game_info.get('referee'),
                "home_club_manager": game_info.get('home_manager', {}).get('name'),
                "away_club_manager": game_info.get('away_manager', {}).get('name'),
                "home_club_formation": game_info.get('home_formation', '').replace("Starting Line-up: ", ""),
                "away_club_formation": game_info.get('away_formation', '').replace("Starting Line-up: ", "")
            }
            if new_game['game_id'] in game_ids_seen:
                continue
            game_ids_seen.append(new_game['game_id'])
            new_game['iso_date'] = parser.parse(new_game.get('date',"1998") or "1998")
            
            for event_num, game_event in enumerate(game_info.get('events', [])):
                if game_event.get('club', {}).get('href', '') == new_game['home_club_href']:
                    new_game['home_team'] = game_event.get('club', {}).get('name')
                elif game_event.get('club', {}).get('href', '') == new_game['away_club_href']:
                    new_game['away_team'] = game_event.get('club', {}).get('name')
                new_game_event = {"game_id": new_game["game_id"], "event_num": event_num, "iso_date": new_game['iso_date']}
                for event_attribute in ['type', 'minute', 'extra', 'action']:
                    new_game_event[event_attribute] = game_event.get(event_attribute)
                new_game_event['club'] = game_event.get('club', {}).get('name')
                new_game_event['player_href'] = game_event.get('player', {}).get('href')
                new_events.append(new_game_event)
            if 'home_team' not in new_game:
                new_game['home_team'] = translate_href_to_name(new_game['home_club_href'])
            if 'away_team' not in new_game:
                new_game['away_team'] = translate_href_to_name(new_game['away_club_href'])
            for team, suffix, l_type in [("home", "team", "starter"), ("away", "team", "starter"), ("home", "bench", "bench"), ("away" ,"bench", "bench")]:
                new_lineup = {
                    "iso_date": new_game['iso_date'],
                    "game_id": new_game['game_id'],
                    "lineup_type": l_type,
                    "team": new_game[f"{team}_team"]
                }
                lineup_info = game_object.get('lineup', {}).get('positions', {}).get(f"{team}_{suffix}", {})
                for position, position_players in lineup_info.items():
                    positions_values = [s.replace('\u20ac', '') for s in position_players if s != '-']
                    for pv in positions_values:
                        factor, exp = pv[0:-1], pv[-1].lower()
                        actual_value = int(float(factor) * TENS[exp])
                        new_lineup.setdefault(position, []).append(actual_value)
                new_lineups.append(new_lineup)
            new_games.append(new_game)
        for new_documents, collection, attribute_keys in [
            (new_games, "tfmkt_games", ['iso_date', 'game_id']), 
            (new_events, "tfmkt_events", ['iso_date', 'game_id', 'event_num']), 
            (new_lineups, "tfmkt_lineups", ['iso_date', 'game_id', 'lineup_type', 'team'])
        ]:
            doc_collection = mongo_connection.get_database('soccer-challenge').get_collection(collection)
            click.echo(f"{len(new_documents)} items updated in {collection}.")
            if upsert:
                outcome = doc_collection.bulk_write(
                    [
                        UpdateOne(
                            filter={"game_id": item['game_id']},
                            update={"$set": {"game_id": item["game_id"]}},
                            upsert=True
                        ) 
                    for item in new_documents]
                )
            else:
                
                doc_collection.insert_many(documents=new_documents)
            
def translate_href_to_name(team_href):
    href_name_part = team_href.split('/')[1]
    return href_name_part.replace('-', ' ')



if __name__ == '__main__':
    cli()
