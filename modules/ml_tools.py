import spacy
import json
import re
import copy
from spacy.lang.en import English
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# nlp = spacy.load("en_core_web_sm")

LEVEL_MAPPING_DIC = {'Internship':0, 'Entry Level':1, 'Mid Level':2, 'Senior Level':3}
INVERT_LVL_DIC = {value: key for key, value in LEVEL_MAPPING_DIC.items()}

def load_spacy_model(func):
    nlp = spacy.load("en_core_web_sm")

    with open("data.json") as f:
        data = json.load(f)
    
    def wrapper(*args, **kwargs):
        args = [data, nlp]
        return func(*args, **kwargs)
    
    return wrapper

@load_spacy_model
def process_text(data, nlp):
    embeddings = generate_embeddings(data, with_lemma=True, nlp = nlp)
    model_data = [e for e in embeddings if e["level"] is not None]
    missing_title_data = [e for e in embeddings if e["level"] is None]
    return embeddings, model_data, missing_title_data

def tokenize_text(string_input, nlp):
    text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', string_input)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]
    return tokens

def generate_embeddings(data, with_lemma=False, nlp=None):
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
    infer_data = copy.deepcopy(embedding_data)

    for e in infer_data:
        infered_lvl_tit = heuristic_rules(e["title"])
        infered_lvl_desc = heuristic_rules(e["desc_text"])

        if infered_lvl_tit is not None:
            e["level_inf"] = infered_lvl_tit
        if infered_lvl_desc is not None:
            e["level_inf"] = infered_lvl_desc
    
    return infer_data

def run_xgboost(X_train, X_test, y_train, y_test):

    model = xgb.XGBClassifier(
        n_estimators=2,  
        max_depth=20,      
        learning_rate=0.03, 
        subsample=0.7,   
        colsample_bytree=0.6,  
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    accuracy_tr = accuracy_score(y_train, y_train_pred)
    y_pred = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred)

    return model, accuracy_tr, accuracy_test
    