import os
from pymongo import MongoClient

def open_mongo_secure_connection():
    """_summary_

    Returns:
        bool: True/False wether the connection was successful.
        : Connection object to MongoDB.
        : Error messages if needed.
    """
    user_id = os.environ.get('MONGODB_USER', None)
    pwd = os.environ.get('MONGODB_PWD', None)
    if user_id is None:
        return False, None, {"msg": "You need to specify your MONGODB_USER in your environment"}
    connection_string = f"mongodb+srv://{user_id}:{pwd}@production.lwhjg.mongodb.net/soccer-challenge?"
    mongo_connection = MongoClient(connection_string)
    return True, mongo_connection, {"msg": "Connection successful"}