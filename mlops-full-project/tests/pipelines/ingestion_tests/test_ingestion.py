import pytest
import pandas as pd
from great_expectations.core import ExpectationSuite
from src.mlops_full_project.pipelines.ingestion.pipeline import ingestion
from src.mlops_full_project.pipelines.ingestion.nodes import build_expectation_suite, fill_na

@pytest.fixture
def sample_dataframe():
    data = {
        "gender": ["Female", "Male", "Unknown/Invalid"],
        "race": ["Caucasian", "Unknown", "AfricanAmerican"],
        "change_in_meds_during_hospitalization": ["Ch", "No", "No"],
        "prescribed_diabetes_meds": ["No", "Yes", "No"],
        "outpatient_visits_in_previous_year": [30, 10, 3],
        "number_lab_tests": [10, 7, 3],
        "number_of_medications": [10, 5, 1],
        "number_diagnoses": [1, 2, 5],
        "average_pulse_bpm": [100, 80, 140]
    }
    return pd.DataFrame(data)

@pytest.fixture
def parameters():
    return {
        "target_column": "target",
        "to_feature_store": False
    }

def test_build_expectation_suite_numerical():
    suite = build_expectation_suite("numerical_expectations", "numerical_features")
    assert isinstance(suite, ExpectationSuite)
    assert len(suite.expectations) > 0

def test_build_expectation_suite_categorical():
    suite = build_expectation_suite("categorical_expectations", "categorical_features")
    assert isinstance(suite, ExpectationSuite)
    assert len(suite.expectations) > 0

def test_build_expectation_suite_target():
    suite = build_expectation_suite("target_expectations", "target")
    assert isinstance(suite, ExpectationSuite)
    assert len(suite.expectations) > 0

def test_fill_na():
    data = pd.DataFrame({
        'country': [None, 'USA'],
        'race': [None, 'Caucasian'],
        'age': [None, 30]
    })
    filled_data = fill_na(data)
    assert filled_data['country'].iloc[0] == 'Unknown'
    assert filled_data['race'].iloc[0] == 'Unknown'
    assert filled_data['age'].iloc[0] == 'Unknown'

def test_ingestion(sample_dataframe, parameters):
    processed_df = ingestion(sample_dataframe, parameters)
    assert 'datetime' in processed_df.columns
    assert processed_df.isnull().sum().sum() == 0  # Check no NaN values
    assert processed_df['gender'].iloc[1] == 'Unknown'  # Check fill_na function
