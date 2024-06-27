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
)

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    ingestion_pipeline = ingestion.create_pipeline()
    data_unit_tests_pipeline = data_unit_tests.create_pipeline()
    split_data_pipeline = split_data.create_pipeline()
    pre_processing_train_pipeline = preprocessing_train.create_pipeline()
    pre_processing_batch_pipeline = preprocessing_batch.create_pipeline()
    split_train_pipeline = split_train.create_pipeline()


    return {
        "ingestion": ingestion_pipeline,
        "data_unit_tests": data_unit_tests_pipeline,
        "split_data": split_data_pipeline,
        "pre_processing_train": pre_processing_train_pipeline,
        "pre_processing_batch": pre_processing_batch_pipeline,
        "split_train": split_train_pipeline
    }
