
import os
import time

if __name__ == "__main__":

    try:
        from multiprocessing import set_start_method
    except ImportError:
        raise ImportError("Unable to import multiprocessing.set_start_method."
                          " This example only runs on Python 3.4")
    set_start_method("forkserver", force=True)

    # import libraries
    import xgboost as xgb
    import pandas as pd

    # get data
    X_train = pd.read_csv('ieee-fraud-detection/train_combined_OHC_xtrain.csv')
    y_train = pd.read_csv('ieee-fraud-detection/train_combined_OHC_ytrain.csv')
    X_test = pd.read_csv('ieee-fraud-detection/train_combined_OHC_xtest.csv')
    y_test = pd.read_csv('ieee-fraud-detection/train_combined_OHC_ytest.csv')

    # set number of threads
    # verified via small experiment to see with how many processors the
    # computer performs best
    num_threads = 4
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    # transform data to xgb matrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # for efficiency
    dtrain.save_binary('train.buffer')

    # parameters
    param = {
        'booster': 'gbtree',  # default
        'verbosity': '2',  # see info messages
        'nthread': num_threads,
        'random_state': 1,

        # Regularization parameters
        # learning rate usually between 0.1 - 0.3
        'eta': 0.3,
        # minimum loss reduction, low since the trees should not be very deep
        'gamma': 0.1,
        # default, could be increased if computational resources increase
        'max_depth': 6,
        # minimum number of observations in a tree, could also be optimized
        'min_child_weight': 400,

        # imbalanced dataset
        'max_delta_step': 5,  # recommendd 1-10 for highly imbalanced dataset
        'scale_pos_weight': 27,  # all_pos/ all_neg (at 3% fraud occurrence)

        # evaluation
        'eval_metric': 'auc',  # Kaggle competition's metric
        'objective': 'binary:logistic'  # for binary classification

        # may use subsampling in future models to avoid over-fitting and
        # increase computation speed
        # 'subsample': 0.8,
        # 'colsample_bynode': 1,
    }

    # validations set to watch performance
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    # training
    num_boost_round = 1000
    bst = xgb.train(param, dtrain, num_boost_round, evallist,
        early_stopping_rounds=20)
    bst.save_model('v1.model')
