
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import yaml
import pickle
import warnings
warnings.filterwarnings("ignore", category=Warning)

# from lib.functions import log_system_metrics

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
from mlflow.tracking import MlflowClient
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

     
def model_selection(X_train: pd.DataFrame, 
                    X_test: pd.DataFrame, 
                    y_train: pd.DataFrame, 
                    y_test: pd.DataFrame,
                    champion_dict: Dict[str, Any],
                    champion_model : pickle.Pickler,
                    parameters: Dict[str, Any]):
    
    
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.DataFrame): Training target.
        y_test (pd.DataFrame): Test target.
        parameters_grid (dict): Parameters defined in parameters_grid.yml.

    Returns:
    --
        models (dict): Dictionary of trained models.
        scores (pd.DataFrame): Dataframe of model scores.
    """

    client = MlflowClient()
   
    models_dict = {
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
    }

    initial_results = {}   

    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        logger.info(experiment_id)


    logger.info('Starting first step of model selection : Comparing between model types')

    # log_system_metrics()

    for model_name, model in models_dict.items():
        with mlflow.start_run(experiment_id=experiment_id,nested=True) as run:
            mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
            y_train = np.ravel(y_train)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"{model_name} Accuracy: {accuracy:.4f}")
            logger.info(f"{model_name} Precision: {precision:.4f}")
            logger.info(f"{model_name} Recall: {recall:.4f}")
            logger.info(f"{model_name} F1-score: {f1:.4f}")

            mlflow.log_metric(f"{model_name} Accuracy", accuracy)
            mlflow.log_metric(f"{model_name} Precision", precision)
            mlflow.log_metric(f"{model_name} Recall", recall)
            mlflow.log_metric(f"{model_name} F1_score", f1)
            # model_name_str = model_name.__class__

            # # Set model version tag
            # client.set_model_version_tag(f"{model_name_str}", "1", "validation_status", "approved")

            initial_results[model_name] = f1
            run_id = mlflow.last_active_run().info.run_id
            logger.info(f"Logged model : {model_name} in run {run_id}, F1-score: {f1}")
    
    best_model_name = max(initial_results, key=initial_results.get)
    best_model = models_dict[best_model_name]

    logger.info(f"Best model is {best_model_name} with F1-score {initial_results[best_model_name]}")
    logger.info('Starting second step of model selection : Hyperparameter tuning')

    # Perform hyperparameter tuning with GridSearchCV
    param_grid = parameters['hyperparameters'][best_model_name]
    with mlflow.start_run(experiment_id=experiment_id,nested=True):
        gridsearch = GridSearchCV(best_model, param_grid, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
        gridsearch.fit(X_train, y_train)
        best_model = gridsearch.best_estimator_


    importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': best_model.feature_importances_})
    importance_df = importance_df.sort_values('importance', ascending=False)
    mlflow.log_table(importance_df, "feature_importances.json")

    # SHAP Values for Explainability
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig("shap_summary.png")
    mlflow.log_artifact("shap_summary.png")

    logger.info(f"Hypertunned model score: {gridsearch.best_score_}")
    pred_score = f1_score(y_test, best_model.predict(X_test))
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"{best_model_name} Accuracy: {accuracy:.4f}")
    logger.info(f"{best_model_name} Precision: {precision:.4f}")
    logger.info(f"{best_model_name} Recall: {recall:.4f}")
    logger.info(f"{best_model_name} F1-score: {f1:.4f}")

    mlflow.log_metric(f"{best_model_name} Accuracy", accuracy)
    mlflow.log_metric(f"{best_model_name} Precision", precision)
    mlflow.log_metric(f"{best_model_name} Recall", recall)
    mlflow.log_metric(f"{best_model_name} F1_score", f1)

    if champion_dict['test_score'] < pred_score:
        logger.info(f"New champion model is {best_model_name} with score: {pred_score} vs {champion_dict['test_score']} ")
        mlflow.sklearn.log_model(best_model, "model")
        model_version = mlflow.register_model(f"runs:/{run.info.run_id}/model", "champion_model")
        mlflow.set_tag("model_description", f"New champion model: {best_model_name}")
        # create "champion" alias for version 1 of model 
        client.set_registered_model_alias(best_model.__class__, "Champion", "1")
        with open("champion_model.pkl", "wb") as model_file:
            pickle.dump(best_model, model_file)
        mlflow.log_artifact("champion_model.pkl")
        return best_model
    else:
        logger.info(f"Champion model is still {champion_dict['classifier']} with score: {champion_dict['test_score']} vs {pred_score} ")
        return champion_model
    
