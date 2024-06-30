import pytest
import pandas as pd
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner
from kedro.io import DataCatalog, MemoryDataSet
from src.mlops_full_project.pipelines.model_selection.pipeline import create_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
import numpy as np
import pickle

@pytest.fixture
def sample_train_data():
    X = pd.DataFrame({
        'inpatient_visits': np.random.rand(120),
        'outpatient_visists': np.random.rand(120)
    })
    y = pd.DataFrame({'readmitted_binary': np.random.choice([0, 1], 120)})
    
    # Split into train and test
    train_size = 100
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test

@pytest.fixture
def champion_dict():
    return {
        "model_name": "RandomForestClassifier",
        "accuracy": 0.85,
        "f1_score": 0.84
    }

@pytest.fixture
def champion_model():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    return pickle.dumps(model)

@pytest.fixture
def sample_test_data():
    data = {
        'inpatient_visits': np.random.rand(20),
        'outpatient_visits': np.random.rand(20),
        'readmitted_binary': np.random.choice([0, 1], 20)
    }
    return pd.DataFrame(data)

@pytest.fixture
def parameters():
    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 20],
        "max_features": [0.5, 0.7, 1],
        "random_state": [19],
        "class_weight": ["balanced", None]
    }

@pytest.fixture
def catalog(sample_data, champion_dict, champion_model, parameters):
    X_train, X_test, y_train, y_test = sample_data
    return DataCatalog({
        "X_train": MemoryDataSet(X_train),
        "X_test": MemoryDataSet(X_test),
        "y_train": MemoryDataSet(y_train),
        "y_test": MemoryDataSet(y_test),
        "champion_dict": MemoryDataSet(champion_dict),
        "champion_model": MemoryDataSet(champion_model),
        "parameters": MemoryDataSet(parameters),
        "best_model": MemoryDataSet()
    })

def test_model_selection_pipeline(catalog):
    pipeline = create_pipeline()
    runner = SequentialRunner()
    result = runner.run(pipeline, catalog)

    assert "best_model" in result
    best_model = result["best_model"]
    
    # Check if the output is a scikit-learn estimator
    assert isinstance(best_model, BaseEstimator)
    
    # Check if the model has necessary methods
    assert hasattr(best_model, 'fit')
    assert hasattr(best_model, 'predict')
    
    assert isinstance(best_model, RandomForestClassifier) or isinstance(best_model, GradientBoostingClassifier) or isinstance(best_model, DecisionTreeClassifier)
    
    X_test = catalog.load("X_test")
    y_test = catalog.load("y_test")
    
    # Check if the model can make predictions
    predictions = best_model.predict(X_test)
    assert len(predictions) == len(y_test)
    
    # Check if the model's score method works
    score = best_model.score(X_test, y_test)
    assert 0 <= score <= 1

    # Compare with champion model
    champion_dict = catalog.load("champion_dict")
    assert score >= champion_dict["accuracy"], "New model should be at least as good as the champion model"

if __name__ == "__main__":
    pytest.main([__file__])

