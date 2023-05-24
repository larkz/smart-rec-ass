### Augment job information
Given the list of jobs (title, description and seniority level),
but some of the jobs are missing the seniority level.

Write an application which fills in the gaps - restores the missing seniority level.

Extra:

## 1. Explain the choice of language / technology stack.

We use Python to perform the prediction model (X -> Y), where X (description) are features in the dataset and Y (seniority + title) are the features.

To address this classification problem we use a combination of the python libraries.
- word heuristics (regex)
- NLP (spacy Doc2Vec)
- supervised ML (xgboost)

The libraries used are in the `requirements.txt` file.

To download the trained word embedding for spacy:

```python -m spacy download en_core_web_sm```

### Model Serving

For serving of the model, I propose the use of Docker + Kubernetes to deploy the NLP algorithm at scale. Some considerations to make during model deployment:

- **Data Transformation (ETL):** This pipeline involves using the transform_data and preprocess_data functions to convert raw data into a suitable format for machine learning. 

- **Model Training:**  In production we need to concern ourselves with the size of the data as well. Sometimes we can scale down the problem by sampling, or shifting the lifting of big data to feature creation instead of ML training. Here we just have a small example.

- **Reuse, Modularity and Reproducibility:** The data pipeline is designed to be reusable by both the ML models. This practice is beneficial as we can avoid the need to re-transform data each time we run a machine learning pipeline. By using the same preprocessed data, we can train multiple machine learning models.

- **Metrics and Evaluation:** Metrics are automatically computed and evaluated during the machine learning pipeline. This should also be the case in production to maintain the integrity of the machine learning model and to facilitate continuous improvement.

## 2. Explain the choice of approach and algorithm.

Goal: predict seniority level

### Heuristic model

Key pattern detection via regex.

### ML Model
- Doc2Vec to transform text into feature vector.
- Supervised learning to predict feature vector -> label.

We run the model hiearchically, heuristics for where it is possible to detect a keyword in the text. Followed by the ML model

### Heuristics

We use basic heuristic rules in the `heuristic_rules(string_input)` function to apply basic rules for the clasification process. For example if the job description has "senior" or "vp" contained in the string, then the seniority level of the job is very likely a senior role. We use regex implement this categorization.

### Word Embedding

We apply the Spacy Doc2Vec word embedding to map the string in the description property to a numeric vector, via the `en_core_web_sm` word embedding.

### Supervised learning


## 3. Estimate quality of the result.

We first estimate the overall accuracy. That is the coorectly identified labels via the algorithm,

We should also have other considerations:
Consideration of robustness (stability)
Generalizability.


```
model = xgb.XGBClassifier(
    n_estimators=3,  
    max_depth=99,       
    learning_rate=0.03, 
    subsample=0.7,     
    colsample_bytree=0.6,  
)
```


