
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import run_data_unit_tests

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_data_unit_tests,
                inputs="ingested_data",
                outputs="data_unit_tests_report",
                name="data_unit_tests",
            ),
        ]
    )
