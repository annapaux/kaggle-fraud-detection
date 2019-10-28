import xgboost as xgb
import pandas as pd

print('start booster')
booster = xgb.Booster()
print('load_model')
booster.load_model('/kaggle-fraud/v1.model')

print('read data')
X_raw = pd.read_csv('/kaggle-fraud/ieee-fraud-detection/test_combined_OHC.csv')
print('transform data')
X_test = xgb.DMatrix(X_raw)
print('predict data')
Y_test = booster.predict(X_test)
print('write predictions')
pd.DataFrame(Y_test).to_csv(r'/kaggle-fraud/ieee-fraud-detection/v1_pred.csv', index=None, header=True)
