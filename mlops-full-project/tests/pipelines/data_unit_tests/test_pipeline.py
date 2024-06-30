import pytest
from kedro.runner import SequentialRunner
from kedro.io import DataCatalog, MemoryDataSet
from src.mlops_full_project.pipelines.data_unit_tests.pipeline import create_pipeline
import pandas as pd

# Fixture to provide sample data for testing
@pytest.fixture
def sample_dataframe():
    data = {
        "gender": ["Male", "Female", "Unknown/Invalid"],
        "race": ["Caucasian", "AfricanAmerican", "Unknown"],
        "change_in_meds_during_hospitalization": ["No", "Ch", "Ch"],
        "prescribed_diabetes_meds": ["Yes", "No", "Yes"],
        "outpatient_visits_in_previous_year": [2, 1, 3],
        "number_lab_tests": [5, 7, 4],
        "number_of_medications": [3, 4, 2],
        "number_diagnoses": [1, 2, 1],
        "average_pulse_bpm": [70, 80, 60]
    }
    return pd.DataFrame(data)

# Fixture to provide a DataCatalog for the test
@pytest.fixture
def catalog(sample_dataframe):
    return DataCatalog({
        "input_data": MemoryDataSet(data=sample_dataframe),
        "output_data": MemoryDataSet()
    })

# Fixture to provide a Kedro runner for the test
@pytest.fixture
def runner():
    return SequentialRunner()

# Test the pipeline execution
def test_pipeline_execution(runner, catalog):
    pipeline = create_pipeline()
    result = runner.run(pipeline, catalog)

    assert "output_data" in result
    output_data = result["output_data"]
    
    # Add assertions to validate the output data
    assert isinstance(output_data, pd.DataFrame)
    assert not output_data.empty

    # Check that specific columns are present in the output
    expected_columns = [
        "gender", "race", "change_in_meds_during_hospitalization", 
        "prescribed_diabetes_meds", "outpatient_visits_in_previous_year",
        "number_lab_tests", "number_of_medications", "number_diagnoses", "average_pulse_bpm"
    ]
    for column in expected_columns:
        assert column in output_data.columns

if __name__ == "__main__":
    pytest.main([__file__])
