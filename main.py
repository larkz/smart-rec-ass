import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from modules.ml_tools import *

# Generate and save embeddings, 
embeddings, model_data, oos_data = process_text(spacy_lib_str="en_core_web_sm")
# In sample inference from heuristic rules
in_samp_data = infer_lvl_from_rules(model_data)
model, accuracy_tr, accuracy_test = run_xgboost_pipeline(in_samp_data)

print(f"XGB Training Accuracy: {accuracy_tr:.2f}")
print(f"XGB Testing Accuracy: {accuracy_test:.2f}")

in_samp_acc = in_sample_accuracy(in_samp_data, model)

print(f"Hiearchical Model In-Sample Accuracy (full coverage): {in_samp_acc:.2f}")

# Out of sample data
infer_data = run_inference_oos(oos_data, model)

with open('output/inference_output.json', 'w') as file:
    json.dump(infer_data, file)

print("Inference results saved to 'output/inference_output.json'.")
