import csv
import click
import os

from tqdm import tqdm
from dateutil import parser
from scripts.scrapping_ops import TOURNAMENTS_OP

odds_portal_folder = "input/odds_portal_files"
countries = ['SPA1']

list_of_files = os.listdir(odds_portal_folder)

def upload_data_file(file_path, country_code):
    file_obj = open(file_path, 'r')
    reader = csv.DictReader(file_obj)
    matches_list = []
    for row in reader:
        row['iso_date'] = parser.parse(row['match_date'])
        row['Div'] = country_code
        convert_to_numeric(
            row,
            excluded_keys=['Div', 'iso_date', 'home_team', 'away_team']
        )
        matches_list.append(row)


for country_code, tournament_info in TOURNAMENTS_OP.items():
    if country_code in countries:
        files_to_upload = [file_name for file_name in list_of_files if file_name.startswith(country_code)]
    for data_file in tqdm(files_to_upload):
        upload_data_file(os.path.join(odds_portal_folder, data_file), country_code=country_code)

files_to_upload

from utils import mongo_db
from utils import dataset

from pymongo.operations import ReplaceOne, UpdateOne

s, mongo_connection, msg = mongo_db.open_mongo_secure_connection()
op_collection = mongo_connection.get_database('soccer-challenge').get_collection('odds_portal_matches')
training_collection = mongo_connection.get_database('soccer-challenge').get_collection('training_matches')
results = op_collection.aggregate([
    {"$group": {
        "_id": {"Sea": "$Sea", "Lge": "$Lge"},
        "total_count": {"$count": {}} 
    }},
    {"$sort": {"_id.Lge": 1, "_id.Sea": 1}}
])

op_results = [r for r in results]


results = training_collection.aggregate([
    {"$group": {
        "_id": {"Sea": "$Sea", "Lge": "$Lge"},
        "total_count": {"$count": {}} 
    }},
    {"$sort": {"_id.Lge": 1, "_id.Sea": 1}}
])

training_results = [r for r in results]