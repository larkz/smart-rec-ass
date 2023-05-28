## Augment job information
Given the list of jobs (title, description and seniority level),
but some of the jobs are missing the seniority level.

Write an application which fills in the gaps - restores the missing seniority level.

Extra:

## 1. Explain the choice of language / technology stack.

We have chosen Python as the language for developing the prediction model (X -> Y), where X represents the description features in the dataset and Y represents the seniority and title features.

To address this classification problem, we utilize the following Python libraries:

- Word heuristics (regex)
- Natural Language Processing (NLP) with Spacy's Doc2Vec
- Supervised Machine Learning with XGBoost

The required libraries are listed in the `requirements.txt` file.

To download the trained word embedding for Spacy, run the following command:

```bash
python -m spacy download en_core_web_sm
```

### Model Serving

For serving of the model, I propose the use of Docker + Kubernetes to deploy the NLP algorithm at scale. Some considerations to make during model deployment:

- **Data Transformation (ETL):** This pipeline involves using the transform_data and preprocess_data functions to convert raw data into a suitable format for machine learning. 

- **Model Training:**  In production we need to concern ourselves with the size of the data as well. Sometimes we can scale down the problem by sampling, or shifting the lifting of big data to feature creation instead of ML training. Here we just have a small example.

- **Reuse, Modularity and Reproducibility:** The data pipeline is designed to be reusable by both the ML models. This practice is beneficial as we can avoid the need to re-transform data each time we run a machine learning pipeline. By using the same preprocessed data, we can train multiple machine learning models.

- **Metrics and Evaluation:** Metrics are automatically computed and evaluated during the machine learning pipeline. This should also be the case in production to maintain the integrity of the machine learning model and to facilitate continuous improvement.

## 2. Explain the choice of approach and algorithm.

Goal: predict seniority level

### Heuristic model

We employ key pattern detection using regular expressions (regex) to implement a heuristic model.

### ML Model
- Doc2Vec: We utilize Spacy's Doc2Vec to transform text into a feature vector.
- Supervised Learning: The ML model performs supervised learning to predict the feature vector and corresponding label.

We run the model hierarchically, applying heuristics where possible to detect specific keywords in the text. Subsequently, the ML model performs supervised learning from a word embedding vector to classify the job seniority level.

We apply `XGBClassifier` to perform the supervised learning.

```
model = xgb.XGBClassifier(
    n_estimators=3,  
    max_depth=99,       
    learning_rate=0.03, 
    subsample=0.7,     
    colsample_bytree=0.6,  
)
```

### Heuristics

We implement basic heuristic rules within the `heuristic_rules(string_input)` function to support the classification process. For example, if the job description contains the terms "senior" or "vp," it is highly likely that the seniority level of the job is senior. We employ regex to implement these categorizations.

### Word Embedding

We utilize Spacy's Doc2Vec word embedding to map the description property to a numeric vector using the `en_core_web_md` word embedding. The model is trained using a shallow parser, a POS tagger, and word vectors trained on Common Crawl data. These include tokenization, part-of-speech (POS) tagging, dependency parsing, named entity recognition (NER), and more.

Other libraries can include,
- `en_core_web_sm`
- `en_core_web_lg`
- `en_core_web_trf`

### Supervised learning


## 3. Estimate quality of the result.

We first estimate the overall accuracy by evaluating the correctly identified labels through the algorithm.

Other considerations include:

- Robustness and Stability
- Generalizability

To improve the performance of the algorithm, the following steps can be taken:

- Increase the size of the training corpus.
- Utilize a custom corpus tailored to the specific domain.

### Test Driven Development

To run unit tests

```
pytest tests/unit_tests.py
```