"""Project pipelines."""
from __future__ import annotations
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from mlops_full_project.pipelines import (
    pre_processing
)

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pre_processing_pipeline = pre_processing.create_pipeline()
    # feature_selection_pipeline = feature_selection.create_pipeline()

    return {
        "pre_processing": pre_processing_pipeline
    }
