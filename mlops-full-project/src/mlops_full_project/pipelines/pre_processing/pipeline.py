
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [


            node(
                func= clean_data,
                inputs="dataset",
                outputs= "data_cleaned",
                name="clean_data",
            ),
            # node(
            #     func= feature_engineer,
            #     inputs="ref_data_cleaned",
            #     outputs= ["preprocessed_training_data","encoder_transform"],
            #     name="preprocessed_training",
            # ),

        ]
    )
