import spacy
import json
import re
import copy
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy as np
from common.variables import *

def load_spacy_model(func):
    """
    Decorator function to load a Spacy model from a file before executing the decorated function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """
    with open("data.json") as f:
        data = json.load(f)
    
    def wrapper(*args, **kwargs):
        args = [data]
        return func(*args, **kwargs)
    
    return wrapper

@load_spacy_model
def process_text(data, spacy_lib_str = "en_core_web_md"):
    """
    Process the given text data using Spacy.

    Args:
        data (list): The data to be processed.
        spacy_lib_str (str): The Spacy library string specifying the model to be loaded (default: "en_core_web_md").

    Returns:
        tuple: A tuple containing embeddings, model data, and missing title data.
    """

    nlp = spacy.load(spacy_lib_str)

    embeddings = generate_embeddings(data, with_lemma=True, nlp = nlp)
    model_data = [e for e in embeddings if e["level"] is not None]
    missing_title_data = [e for e in embeddings if e["level"] is None]
    return embeddings, model_data, missing_title_data

def tokenize_text(string_input, nlp):
    """
    Tokenize the given string using Spacy.

    Args:
        string_input (str): The string to be tokenized.
        nlp: The Spacy model.

    Returns:
        list: A list of tokens.
    """
    text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', string_input)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]
    return tokens

def generate_embeddings(data, with_lemma=False, nlp=None):
    """
    Generate embeddings for the given data using Spacy.

    Args:
        data (list): The data for which embeddings are to be generated.
        with_lemma (bool): Flag indicating whether to include lemma in the embeddings (default: False).
        nlp: The Spacy model.

    Returns:
        list: A list of embeddings.
    """
    embeddings = []
    for dic in data:
        level = LEVEL_MAPPING_DIC[dic["level"]] if "level" in dic.keys() else None
        title = dic["title"] if "title" in dic.keys() else None

        if with_lemma:
            tokens = tokenize_text(dic["description"].lower(), nlp)
            text_with_sentences = " ".join(tokens).replace(" .", ". ")
            doc = nlp(text_with_sentences) 
        else:
            doc = nlp(dic["description"])
        
        embeddings.append({"level_inf": None, "level": level, "title": title, "desc_text": dic["description"], "doc_vec": doc})
    return embeddings

def heuristic_rules(string_input):
    """
    Apply heuristic rules to determine the level from the given string.

    Args:
        string_input (str): The input string.

    Returns:
        str: The inferred level.
    """
    string = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', string_input)
    
    if re.search(r"senior\b", string, re.UNICODE | re.IGNORECASE) is not None:
        return LEVEL_MAPPING_DIC['Senior Level']
    if re.search(r"sr.\b", string, re.UNICODE | re.IGNORECASE) is not None:
        return LEVEL_MAPPING_DIC['Senior Level']
    if re.search(r"vp\b",  string, re.UNICODE | re.IGNORECASE) is not None:
        return LEVEL_MAPPING_DIC['Senior Level']
    if re.search(r"head of\b", string, re.UNICODE | re.IGNORECASE) is not None:
        return LEVEL_MAPPING_DIC['Senior Level']

    if re.search(r"junior\b", string, re.UNICODE | re.IGNORECASE) is not None:
        return LEVEL_MAPPING_DIC['Entry Level']
    
    if re.search(r"internship\b", string, re.UNICODE | re.IGNORECASE) is not None:
        return LEVEL_MAPPING_DIC['Internship']
    if re.search(r"intern\b", string, re.UNICODE | re.IGNORECASE) is not None:
        return LEVEL_MAPPING_DIC['Internship']
    
    return None

def infer_lvl_from_rules(embedding_data):
    """
    Infer the level from the embedding data using heuristic rules.

    Args:
        embedding_data (list): The embedding data.

    Returns:
        list: The inferred embedding data.
    """
    infer_data = copy.deepcopy(embedding_data)

    for e in infer_data:
        infered_lvl_tit = heuristic_rules(e["title"])
        infered_lvl_desc = heuristic_rules(e["desc_text"])

        if infered_lvl_tit is not None:
            e["level_inf"] = infered_lvl_tit
        if infered_lvl_desc is not None:
            e["level_inf"] = infered_lvl_desc
    
    return infer_data

def run_xgboost(X_train, X_test, y_train, y_test ):
    """
    Run the XGBoost classifier on the given data.

    Args:
        X_train: The training data.
        X_test: The test data.
        y_train: The training labels.
        y_test: The test labels.

    Returns:
        tuple: A tuple containing the trained model, accuracy on training data, and accuracy on test data.
    """
    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,  
        max_depth=MAX_DEPTH,      
        learning_rate=LR, 
        subsample=SUBSAMPLE,   
        colsample_bytree=COLSAMP,
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    accuracy_tr = accuracy_score(y_train, y_train_pred)
    y_pred = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred)

    return model, accuracy_tr, accuracy_test

def run_xgboost_pipeline(in_samp_data):
    """
    Run the XGBoost pipeline on the given data.

    Args:
        in_samp_data: The input data.

    Returns:
        tuple: A tuple containing the trained model, accuracy on training data, and accuracy on test data.
    """

    in_samp_data_x = np.array([d["doc_vec"].vector for d in in_samp_data])
    in_samp_data_y = np.array([d["level"] for d in in_samp_data])
    X_train, X_test, y_train, y_test = train_test_split(in_samp_data_x, in_samp_data_y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    model, accuracy_tr, accuracy_test = run_xgboost(X_train, X_test, y_train, y_test )

    return model, accuracy_tr, accuracy_test

def in_sample_accuracy(in_samp_data, model):
    # In sample accuracy check
    for d in in_samp_data:
        if d['level_inf'] is None:
            d['level_inf'] = model.predict(np.array([d["doc_vec"].vector]))[0]

    # hiearchical model accuracy on in sample data
    data_insamp_correct = [i for i in in_samp_data if i["level"] == i["level_inf"]]
    
    return len(data_insamp_correct)/len(in_samp_data)

def run_inference_oos(oos_data, ml_model):
    
    infer_data = infer_lvl_from_rules(oos_data)
    infer_data_copy = copy.deepcopy(infer_data)

    for d in infer_data_copy:
        
        if d['level_inf'] is None:
            lvl_index = ml_model.predict(np.array([d["doc_vec"].vector]))[0]
            d['level_inf'] = INVERT_LVL_DIC[lvl_index]
            
        # d["doc_vec"] = d["doc_vec"].to_dict()
        if d['level'] == None:
            del d['level']
        del d['doc_vec']

    return infer_data_copy