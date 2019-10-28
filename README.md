# Fraud Detection with XGBoost

## The model
I'm using the XGBoost library to train a gradient boosting tree to classify
what transactions are fraudulent. The dataset has transaction and individual
based observations, which are combined via the TransactionID.

After one-hot-encoding categorical variables and combining the data, there are
up to 1773 features. The training data has 395,662 observations, and the test
set has 194,880.

The model's auc score on the training data was 0.94.

## Data Source
This model is based on a challenge on Kaggle, using a dataset from IEEE.
https://www.kaggle.com/c/ieee-fraud-detection/data

## Files
- process_data.py - process the data to combine identity and transaction and
one hot encode categorical variables
- v1_model.py - specifies the model parameters and saves the model in v1.model
- v1.model - the classification model specified by the XGBoost framework in the
v1_model.py file
- v1_prediction.py - the script to make predictions based on test data and the
model (v1_model.py)
- fraud_data_exploration.ipynb - a jupyter notebook with very basic data
exploration
