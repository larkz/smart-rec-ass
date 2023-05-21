import spacy
import json
import re
import copy
from spacy.lang.en import English

nlp = spacy.load("en_core_web_sm")

LEVEL_MAPPING_DIC = {'Internship':0, 'Entry Level':1, 'Mid Level':2, 'Senior Level':3}
INVERT_LVL_DIC = {value: key for key, value in LEVEL_MAPPING_DIC.items()}

def tokenize_text(text):
    # Create a Doc object by processing the text with the spaCy model
    doc = nlp(text)
    # Extract the tokens from the Doc object and return them as a list
    tokens = [token.text for token in doc]
    return tokens

def generate_embeddings(data):
    embeddings = []
    for dic in data:
        level = LEVEL_MAPPING_DIC[dic["level"]] if "level" in dic.keys() else None
        title = dic["title"] if "title" in dic.keys() else None

        desc_tokens = tokenize_text(dic["description"])
        doc = nlp(dic["description"])
        embeddings.append({"level_inf": None, "level": level, "title": title, "desc_text": dic["description"], "doc_vec": doc.vector})
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



# What is the coverage if we apply heuristic rules?

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