import os
import json

DATA_DIR = "user_data"

def save_user_data(user_id, data):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, f"{user_id}.json"), "w") as f:
        json.dump(data, f)

def load_user_data(user_id):
    try:
        with open(os.path.join(DATA_DIR, f"{user_id}.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"expenses": [], "portfolio": {}}