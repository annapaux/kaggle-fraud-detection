'''
Code to pre-process one hot encoding and combined data and save in csv files

The generated data can be called with the following commands:
X_train = pd.read_csv('ieee-fraud-detection/train_combined_OHC_xtrain.csv')
y_train = pd.read_csv('ieee-fraud-detection/train_combined_OHC_ytrain.csv')
X_test = pd.read_csv('ieee-fraud-detection/train_combined_OHC_xtest.csv')
y_test = pd.read_csv('ieee-fraud-detection/train_combined_OHC_ytest.csv')
'''


# combine transaction and identity + write to csv
transaction = pd.read_csv('ieee-fraud-detection/train_transaction.csv')
identity = pd.read_csv('ieee-fraud-detection/train_identity.csv')
data = pd.merge(transaction, identity, left_on="TransactionID", right_on="TransactionID", how="left")
data.to_csv(r'ieee-fraud-detection/train_combined.csv', index=None, header=True)


# Convert to One Hot Encoding + write to csv
categorical = []
for column in list(data):
    if data[str(column)].dtype == 'object':
        categorical.append(column)

for column in categorical:
    data = pd.concat(
        [data, pd.get_dummies(data[column], prefix=column)], axis=1
        ).drop([column], axis=1)

data.to_csv(r'ieee-fraud-detection\train_combined_OHC.csv', index=None, header=True)


# Train - test split & to csv
from sklearn.model_selection import train_test_split
X = data.loc[:, data.columns != 'isFraud']
y = data['isFraud']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y)

X_train.to_csv(r'ieee-fraud-detection/train_combined_OHC_xtrain.csv', index=None, header=True)
X_test.to_csv(r'ieee-fraud-detection/train_combined_OHC_xtest.csv', index=None, header=True)
y_train.to_csv(r'ieee-fraud-detection/train_combined_OHC_ytrain.csv', index=None, header=True)
y_test.to_csv(r'ieee-fraud-detection/train_combined_OH_.ytest.csv', index=None, header=True)
