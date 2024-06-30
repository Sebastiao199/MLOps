import pytest
import pandas as pd
import great_expectations as gx
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from great_expectations.data_context import DataContext
from src.mlops_full_project.pipelines.data_unit_tests.nodes import get_validation_results, run_data_unit_tests

@pytest.fixture
def checkpoint_result():
    return {
        "run_results": {
            "some_key": {
                "validation_result": {
                    "results": [
                        {
                            "success": True,
                            "expectation_config": {
                                "expectation_type": "expect_column_values_to_be_between",
                                "kwargs": {
                                    "column": "age",
                                    "min_value": 0,
                                    "max_value": 100
                                }
                            },
                            "result": {
                                "element_count": 10,
                                "unexpected_count": 0,
                                "unexpected_percent": 0,
                                "observed_value": [25, 35, 45]
                            }
                        },
                        {
                            "success": False,
                            "expectation_config": {
                                "expectation_type": "expect_column_values_to_be_in_set",
                                "kwargs": {
                                    "column": "gender",
                                    "value_set": ["Male", "Female"]
                                }
                            },
                            "result": {
                                "element_count": 10,
                                "unexpected_count": 2,
                                "unexpected_percent": 20,
                                "observed_value": ["Male", "Female", "Unknown"]
                            }
                        }
                    ],
                    "meta": {
                        "expectation_suite_name": "Hospital"
                    }
                }
            }
        }
    }

def test_get_validation_results_success(checkpoint_result):
    df_validation = get_validation_results(checkpoint_result)
    
    assert not df_validation.empty
    assert "Success" in df_validation.columns
    assert "Expectation Type" in df_validation.columns
    assert df_validation["Success"].iloc[0] == True
    assert df_validation["Expectation Type"].iloc[0] == "expect_column_values_to_be_between"
    assert df_validation["Column"].iloc[0] == "age"

def test_get_validation_results_failure(checkpoint_result):
    df_validation = get_validation_results(checkpoint_result)
    
    assert not df_validation.empty
    assert df_validation["Success"].iloc[1] == False
    assert df_validation["Expectation Type"].iloc[1] == "expect_column_values_to_be_in_set"
    assert df_validation["Column"].iloc[1] == "gender"
    assert df_validation["Unexpected Count"].iloc[1] == 2
    assert df_validation["Unexpected Percent"].iloc[1] == 20
    assert "Unknown" in df_validation["Observed Value"].iloc[1]

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

def test_run_data_unit_tests(sample_dataframe):
    df_validation = run_data_unit_tests(sample_dataframe)
    
    assert isinstance(df_validation, pd.DataFrame)
    assert not df_validation.empty

    # Check the presence of all required columns
    required_columns = ["Success", "Expectation Type", "Column", "Column Pair", "Max Value", "Min Value",
                        "Element Count", "Unexpected Count", "Unexpected Percent", "Value Set", 
                        "Unexpected Value", "Observed Value"]
    for column in required_columns:
        assert column in df_validation.columns

def test_expectations(sample_dataframe):
    context = gx.get_context(context_root_dir="//..//..//gx")
    suite = context.get_expectation_suite("Hospital")
    
    # Check that expectations are added correctly
    assert suite.expectations[0].expectation_type == "expect_column_distinct_values_to_be_in_set"
    assert suite.expectations[0].kwargs["column"] == "gender"
    assert "Male" in suite.expectations[0].kwargs["value_set"]

    assert suite.expectations[1].expectation_type == "expect_column_distinct_values_to_be_in_set"
    assert suite.expectations[1].kwargs["column"] == "race"
    assert "Caucasian" in suite.expectations[1].kwargs["value_set"]

def test_data_types(sample_dataframe):
    pd_df_ge = gx.from_pandas(sample_dataframe)
    
    assert pd_df_ge.expect_column_values_to_be_of_type("average_pulse_bpm", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("number_diagnoses", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("number_of_medications", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("number_lab_tests", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("outpatient_visits_in_previous_year", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("gender", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("race", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("change_in_meds_during_hospitalization", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("prescribed_diabetes_meds", "str").success == True

def test_column_count(sample_dataframe):
    pd_df_ge = gx.from_pandas(sample_dataframe)
    
    # raw dataset file should have 33 columns
    # included 2 columns (index and datetime) after ingestion step
    assert pd_df_ge.expect_table_column_count_to_equal(33).success == True

if __name__ == "__main__":
    pytest.main([__file__])
