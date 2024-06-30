
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import pickle

import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


logger = logging.getLogger(__name__)

def model_predict(X_test: pd.DataFrame,  
                  y_test: pd.DataFrame,
                  model: pickle.Pickler, columns: pickle.Pickler) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Predict using the trained model.

    Args:
    --
        X_test (pd.DataFrame): Test features.
        y_test (pd.DataFrame): Test target.
        model (pickle): Trained model.

    Returns:
    --
        scores (pd.DataFrame): Dataframe with new predictions.
    """

    # Predict
    
    y_pred = model.predict(X_test[columns])

    # Create dataframe with predictions
    X_test['y_pred'] = y_pred


    # Score distribution

    y_pred_probs = model.predict_proba(X_test[columns])

    plt.hist(y_pred_probs[:, 1], bins=20, edgecolor='black')
    plt.xlabel('Predicted Probability (Class 1)')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.savefig("score_distribution.png")

    mlflow.log_artifact("score_distribution.png")


    # Estimated performance

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-score: {f1:.4f}")

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1_score", f1)
        
    # Create dictionary with predictions
    describe_servings = X_test.describe().to_dict()

    logger.info('Service predictions created.')
    logger.info('#servings: %s', len(y_pred))

    return X_test, describe_servings, plt