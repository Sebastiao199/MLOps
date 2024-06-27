
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_engineer, clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [   
            node(
                func= clean_data,
                inputs="ana_data",
                outputs= ["ana_data_cleaned","reporting_data_test"],
                name="clean_data_test",
            ),
            node(
                func= feature_engineer,
                inputs=["ana_data_cleaned","encoder_transform","age_imputer_model"],
                outputs= "preprocessed_batch_data",
                name="preprocessed_batch",
            ),

        ]
    )
