### Augment job information
Given the list of jobs (title, description and seniority level),
but some of the jobs are missing the seniority level.

Write an application which fills in the gaps - restores the missing seniority level.

Extra:

1. Explain the choice of language / technology stack.

ETL?
Python, for ML
Serving?

- **Data Transformation (ETL):** This pipeline involves using the transform_data and preprocess_data functions to convert raw data into a suitable format for machine learning. For this exercise, we utilize scikit-learn to train the machine learning model.

- **Model Training:** Subsequently, we implement model training for the RF or LR model via scikit-learn. In production we need to concern ourselves with the size of the data as well. Sometimes we can scale down the problem by sampling, or shifting the lifting of big data to feature creation instead of ML training. Here we just have a small example.

- **Reuse, Modularity and Reproducibility:** The data pipeline is designed to be reusable by both the ML models. This practice is beneficial as we can avoid the need to re-transform data each time we run a machine learning pipeline. By using the same preprocessed data, we can train multiple machine learning models.

- **Metrics and Evaluation:** Metrics are automatically computed and evaluated during the machine learning pipeline. This should also be the case in production to maintain the integrity of the machine learning model and to facilitate continuous improvement.

2. Explain the choice of approach and algorithm.

Goal: predict seniority level

Doc2Vec to transform text into feature vector.
Supervised learning to predict feature vector -> label.

3. Estimate quality of the result.

Consideration of robustness (stability)
Generalizability.

# python -m spacy download en_core_web_sm
