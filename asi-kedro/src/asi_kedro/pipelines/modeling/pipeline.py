"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_data_for_modeling, split_data, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_data_for_modeling,
                inputs="CarPrice",
                outputs="data_prepared_for_modeling",
                name="prepare_data_node",
            ),
            node(
                func=split_data,
                inputs="data_prepared_for_modeling",
                outputs=["xtrain", "xtest", "ytrain", "ytest"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["xtrain", "ytrain"],
                outputs="classifier",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["classifier", "xtest", "ytest"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )
