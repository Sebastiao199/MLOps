"""Project pipelines."""
from __future__ import annotations
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from mlops_full_project.pipelines import (
    feature_selection
)

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    feature_selection_pipeline = feature_selection.create_pipeline()

    return {
        "feature_selection": feature_selection_pipeline
    }
