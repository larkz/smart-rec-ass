import spacy
import json
import re
import copy
import xgboost as xgb
from modules.ml_tools import *

with open("data.json") as f:
    data = json.load(f)

embeddings = generate_embeddings(data)

model_data = [e for e in embeddings if e["level"] is not None]
missing_title_data = [e for e in embeddings if e["level"] is None]

