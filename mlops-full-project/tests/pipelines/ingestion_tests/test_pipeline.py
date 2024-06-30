import pytest
import pandas as pd
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner
from kedro.io import DataCatalog, MemoryDataSet
from src.mlops_full_project.pipelines.ingestion.pipeline import create_pipeline

@pytest.fixture
def sample_dataframe():
    data = {

        'outpatient_visits_in_previous_year': [1, 2, 20],
        'race': ["Unknown", 'Caucasian', 'AfricanAmerican'],
        'gender': ['Male', 'Unknown/Invalid'],
        'target': ['No', 'Yes', 'Yes'],
        "change_in_meds_during_hospitalization": ["No", "Ch", "Ch"],
        "prescribed_diabetes_meds": ["Yes", "No", "Yes"],
        "outpatient_visits_in_previous_year": [2, 1, 3],
        "number_lab_tests": [1, 5, 9],
        "number_of_medications": [3, 9, 11],
        "number_diagnoses": [1, 2, 3]
    }
    return pd.DataFrame(data)

@pytest.fixture
def parameters():
    return {
        "target_column": "target",
        "to_feature_store": False
    }

@pytest.fixture
def catalog(sample_dataframe, parameters):
    return DataCatalog({
        "sample_dataframe": MemoryDataSet(sample_dataframe),
        "parameters": MemoryDataSet(parameters),
        "processed_df": MemoryDataSet()
    })

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

