import spacy
import json
import re
import copy
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pytest

from modules.ml_tools import *

# nlp = spacy.load("en_core_web_sm") # Load pre-trained model

# Generate and save embeddings 
embeddings, model_data, missing_title_data = process_text()

# In sample inference from heuristic rules
in_samp_data = infer_lvl_from_rules(model_data)

# Build xg boost model
in_samp_data_x = np.array([d["doc_vec"].vector for d in in_samp_data])
in_samp_data_y = np.array([d["level"] for d in in_samp_data])
X_train, X_test, y_train, y_test = train_test_split(in_samp_data_x, in_samp_data_y, test_size=0.2, random_state=42)

model, accuracy_tr, accuracy_test = run_xgboost(X_train, X_test, y_train, y_test)
print(accuracy_tr, accuracy_test)




# Out of sample data
# infer_data = infer_lvl_from_rules(missing_title_data)


