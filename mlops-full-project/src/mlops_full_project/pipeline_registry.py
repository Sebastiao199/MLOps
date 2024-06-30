"""Project pipelines."""
from __future__ import annotations
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from mlops_full_project.pipelines import (
    ingestion,
    data_unit_tests,
    split_data,
    preprocessing_train,
    preprocessing_batch,
    split_train_pipeline as split_train,
    feature_selection,
    model_train,
    model_selection,
    model_predict,
    data_drift,
)

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    ingestion_pipeline = ingestion.create_pipeline()
    data_unit_tests_pipeline = data_unit_tests.create_pipeline()
    split_data_pipeline = split_data.create_pipeline()
    preprocessing_train_pipeline = preprocessing_train.create_pipeline()
    preprocessing_batch_pipeline = preprocessing_batch.create_pipeline()
    split_train_pipeline = split_train.create_pipeline()
    feature_selection_pipeline = feature_selection.create_pipeline()
    model_train_pipeline = model_train.create_pipeline()
    model_selection_pipeline = model_selection.create_pipeline()
    model_predict_pipeline = model_predict.create_pipeline()
    data_drift_pipeline = data_drift.create_pipeline()


    return {
        "ingestion": ingestion_pipeline,
        "data_unit_tests": data_unit_tests_pipeline,
        "split_data": split_data_pipeline,
        "preprocessing_train": preprocessing_train_pipeline,
        "preprocessing_batch": preprocessing_batch_pipeline,
        "split_train": split_train_pipeline,
        "feature_selection": feature_selection_pipeline,
        "model_train": model_train_pipeline,
        "model_selection": model_selection_pipeline,
        "model_predict": model_predict_pipeline,
        "data_drift": data_drift_pipeline,
        "production_full_train_process" : preprocessing_train_pipeline + split_train_pipeline + model_train_pipeline,
        "production_full_prediction_process" : preprocessing_batch_pipeline + model_predict_pipeline,
    }
