import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, RFECV
from scipy.stats import pointbiserialr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import os
import pickle


def remove_low_variance_features(data, parameters) -> pd.DataFrame:
    variances = data[parameters["metric_features"]].var()
    low_variance_features = variances[variances <= parameters["variance_threshold"]].index.tolist()
    
    if low_variance_features:
        print(f"Removing low variance features: {low_variance_features}")
        data = data.drop(columns=low_variance_features)
    else:
        print("No low variance features found.")
    
    return data

# def point_biserial_correlation(X, y, parameters):
#     for column in X[parameters["metric_features"]].columns:
#         try:
#             corr, _ = pointbiserialr(X[column], y)
#             if abs(corr) < parameters["point_biserial_threshold"]:
#                 X = X.drop(columns=[column])
#                 print(f"Removed low point biserial correlation feature: {column}")
#         except Exception as e:
#             print(f"Error processing column {column}: {str(e)}")
#             continue
#     return X

# def point_biserial_correlation(X, y, parameters):
#     for column in parameters["metric_features"]:
#         try:
#             # Convert to numeric and handle non-numeric data
#             X[column] = pd.to_numeric(X[column], errors='coerce')
            
#             # Drop rows with NaN values in this column
#             X = X.dropna(subset=[column])
            
#             # Calculate point-biserial correlation
#             corr, _ = pointbiserialr(X[column], y)
#             if abs(corr) < parameters["point_biserial_threshold"]:
#                 X = X.drop(columns=[column])
#                 print(f"Removed low point biserial correlation feature: {column}")
#         except Exception as e:
#             print(f"Error processing column {column}: {str(e)}")
#             continue
#     return X


def mutual_info_selection(X, y, parameters):
    mi_scores = mutual_info_classif(X, y)

    selected_mask = mi_scores > parameters["mi_threshold"]
    selected_features = X.columns[selected_mask].tolist()

    removed_features = X.columns[~selected_mask].tolist()

    if removed_features:
        print(f"Removed low mutual information features: {removed_features}")
    X = X[selected_features]
    return X


def feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]):

    log = logging.getLogger(__name__)
    log.info(f"We start with: {len(X_train.columns)} columns")
    # log.info(X_train.dtypes)
    # log.info(y_train.dtypes)
    # log.info(np.unique(y_train))
    # pd.set_option('display.max_columns', None)
    # log.info(X_train.head(10))


    if "variance_thresholding" in parameters["feature_selection"]:
        remove_low_variance_features(X_train, parameters)

    # if "point_biserial" in parameters["feature_selection"]:
    #     point_biserial_correlation(X_train, y_train, parameters)

    if "mutual_info" in parameters["feature_selection"]:
        mutual_info_selection(X_train, y_train, parameters)

    if "rfe" in parameters["feature_selection"]:
        y_train = np.ravel(y_train)
        # open pickle file with regressors
        try:
            with open(os.path.join(os.getcwd(), 'data', '06_models', 'production_model.pkl'), 'rb') as f:
                classifier = pickle.load(f)
                logging.info("Loaded existing champion model")
        except:
            logging.info("No existing model found. Creating new RandomForestClassifier with baseline parameters.")
            classifier = RandomForestClassifier(**parameters['baseline_model_params'])

        rfe = RFECV(classifier, scoring='f1', n_jobs=-1, cv = StratifiedKFold(5)) 
        rfe = rfe.fit(X_train, y_train)
        f = rfe.get_support(1) #the most important features
        X_cols = X_train.columns[f].tolist()

    log.info(f"Number of best columns is: {len(X_cols)}")
    log.info(f"Best columns: {X_cols}")
    
    return X_cols