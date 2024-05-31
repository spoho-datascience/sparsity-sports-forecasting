import pandas as pd
pd.options.display.max_columns = None
from utils import mongo_db
from utils import dataset

success_in_connection, conn, conn_info = mongo_db.open_mongo_secure_connection()
if success_in_connection:
    print('Yup you are connected')
    print(conn_info)

game_id = 2243026

match_clause = {
    "$match": {
        "game_id": game_id, "training_id": {"$exists": True}
    }
}

tfmkt_games = conn.get_database('soccer-challenge').get_collection('tfmkt_games')
results = tfmkt_games.aggregate([
    match_clause,
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

match_object = next(results)
home_team = match_object.get('home_team', '-')
away_team = match_object.get('away_team', '-')
new_metrics = {}
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





