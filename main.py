import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from modules.ml_tools import *

if __name__ == '__main__':

    # Generate and save embeddings, 
    embeddings, model_data, oos_data = process_text(spacy_lib_str="en_core_web_sm")

    # Hiearchical Stage 1) classifier from heuristic rules 
    in_samp_data = infer_lvl_from_rules(model_data)
    # Train an XGBoost model
    model, accuracy_tr, accuracy_test = run_xgboost_pipeline(in_samp_data)
    print(f"XGB Training Accuracy: {accuracy_tr:.3f}")
    print(f"XGB Testing Accuracy: {accuracy_test:.3f}")

    # Hiearchical 2) Precidt using an XGBoost model
    in_samp_data = run_inference_oos(model_data, model)

    in_samp_acc = in_sample_accuracy(in_samp_data)

    print(f"Hiearchical Model In-Sample Accuracy (full coverage): {in_samp_acc:.3f}")

    # Run inference on out-of-sample data
    infer_data = run_inference_oos(oos_data, model)

    json_data = json.dumps(prepare_json(infer_data), indent=4)

    with open(OUTPUT_FILEPATH, 'w') as file:
        json.dump(json_data, file)

    print(f"Inference results saved to : {OUTPUT_FILEPATH}")
