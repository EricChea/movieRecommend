"""
This module is meant for training a model.  The output is a model that can be
used to make predictions.
"""

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split


RANDOM_STATE = 333


def main():

    import argparse

    from ..utils import data

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="Path to the zip file containing data")
    args = parser.parse_args()

    data_dict = data.read_zipcsv(args.data)
    X, y, _ = data.get_features_dependent(data_dict, 'rating')

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE)
    X_test, X_valid, y_test, y_valid = train_test_split(
        X_valid, y_valid, test_size=0.50, random_state=RANDOM_STATE)

    print("Light Gradient Boosting Regressor")
    lgbm_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 8,
        'num_leaves': 16,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.4,
        'learning_rate': 0.03,
        'verbose': 0
    }

    features = [col for col in X.columns if col not in ('userId', 'movieId')]
    kwargs = dict(
        feature_name=features,
        categorical_feature=[col for col in X.columns if col.endswith('Movie')]
    )

    # LGBM Dataset Formatting
    lgtrain = lgb.Dataset(X_train[features], y_train, **kwargs)
    lgvalid = lgb.Dataset(X_valid[features], y_valid, **kwargs)

    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=16000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train', 'valid'],
        early_stopping_rounds=200,
        verbose_eval=200
    )

    # Feature Importance Plot
    f, ax = plt.subplots(figsize=[7, 10])
    lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
    plt.title("Light GBM Feature Importance")
    plt.savefig('feature_import.png')

    print("Model Evaluation Stage")
    mae = metrics.mean_absolute_error(y_valid, lgb_clf.predict(X_valid[features]))

    print(f"MAE: {mae}")
    lgpred = lgb_clf.predict(X_test[features])
    X_test['yhat_rating'] = lgpred.clip(0.0, 5.0)
    X_test['rating'] = y_test
    X_test.to_csv('test_output.csv', index=False)
    lgb_clf.save_model('trees.predictrating.lgb')


if __name__ == '__main__':
    import sys
    sys.exit(main())
