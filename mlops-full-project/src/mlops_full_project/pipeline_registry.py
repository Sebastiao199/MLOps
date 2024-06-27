"""Project pipelines."""
from __future__ import annotations
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from mlops_full_project.pipelines import (
    #feature_selection,
    pre_processing,
    ingestion,
    data_unit_tests
)

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    #feature_selection_pipeline = feature_selection.create_pipeline()
    pre_processing_pipeline = pre_processing.create_pipeline()
    ingestion_pipeline = ingestion.create_pipeline()
    data_unit_tests_pipeline = data_unit_tests.create_pipeline()

    return {
        #"feature_selection": feature_selection_pipeline,
        "pre_processing": pre_processing_pipeline,
        "ingestion": ingestion_pipeline,
        "data_unit_tests": data_unit_tests_pipeline
    }
