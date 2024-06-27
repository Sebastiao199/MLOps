"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import great_expectations as gx

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

logger = logging.getLogger(__name__)


def get_validation_results(checkpoint_result):
    # validation_result is a dictionary containing one key-value pair
    validation_result_key, validation_result_data = next(iter(checkpoint_result["run_results"].items()))

    # Accessing the 'actions_results' from the validation_result_data
    validation_result_ = validation_result_data.get('validation_result', {})

    # Accessing the 'results' from the validation_result_data
    results = validation_result_["results"]
    meta = validation_result_["meta"]
    use_case = meta.get('expectation_suite_name')
    
    
    df_validation = pd.DataFrame({},columns=["Success","Expectation Type","Column","Column Pair","Max Value",\
                                       "Min Value","Element Count","Unexpected Count","Unexpected Percent","Value Set","Unexpected Value","Observed Value"])
    
    
    for result in results:
        # Process each result dictionary as needed
        success = result.get('success', '')
        expectation_type = result.get('expectation_config', {}).get('expectation_type', '')
        column = result.get('expectation_config', {}).get('kwargs', {}).get('column', '')
        column_A = result.get('expectation_config', {}).get('kwargs', {}).get('column_A', '')
        column_B = result.get('expectation_config', {}).get('kwargs', {}).get('column_B', '')
        value_set = result.get('expectation_config', {}).get('kwargs', {}).get('value_set', '')
        max_value = result.get('expectation_config', {}).get('kwargs', {}).get('max_value', '')
        min_value = result.get('expectation_config', {}).get('kwargs', {}).get('min_value', '')

        element_count = result.get('result', {}).get('element_count', '')
        unexpected_count = result.get('result', {}).get('unexpected_count', '')
        unexpected_percent = result.get('result', {}).get('unexpected_percent', '')
        observed_value = result.get('result', {}).get('observed_value', '')
        if type(observed_value) is list:
            #sometimes observed_vaue is not iterable
            unexpected_value = [item for item in observed_value if item not in value_set]
        else:
            unexpected_value=[]
        
        df_validation = pd.concat([df_validation, pd.DataFrame.from_dict( [{"Success" :success,"Expectation Type" :expectation_type,"Column" : column,"Column Pair" : (column_A,column_B),"Max Value" :max_value,\
                                           "Min Value" :min_value,"Element Count" :element_count,"Unexpected Count" :unexpected_count,"Unexpected Percent":unexpected_percent,\
                                                  "Value Set" : value_set,"Unexpected Value" :unexpected_value ,"Observed Value" :observed_value}])], ignore_index=True)
        
    return df_validation


def run_data_unit_tests(df):
    context = gx.get_context(context_root_dir = "//..//..//gx")
    datasource_name = "dataset"
    try:
        datasource = context.sources.add_pandas(datasource_name)
        logger.info("Data Source created.")
    except:
        logger.info("Data Source already exists.")
        datasource = context.datasources[datasource_name]

    suite_hospital = context.add_or_update_expectation_suite(expectation_suite_name="Hospital")
    
    #add more expectations to your data
    expectation_gender = ExpectationConfiguration(
    expectation_type="expect_column_distinct_values_to_be_in_set",
    kwargs={
        "column": "gender",
        "value_set" : ['Female', 'Male', 'Unknown/Invalid']
    },
    )
    suite_hospital.add_expectation(expectation_configuration=expectation_gender)

    #add more expectations to your data
    expectation_race = ExpectationConfiguration(
    expectation_type="expect_column_distinct_values_to_be_in_set",
    kwargs={
        "column": "race",
        "value_set" : ['Caucasian', 'AfricanAmerican','Unknown', 'Hispanic', 'Other', 'Asian']
    },
    )
    suite_hospital.add_expectation(expectation_configuration=expectation_race)

    #change_in_meds_during_hospitalization
    expectation_change_in_meds_during_hospitalization = ExpectationConfiguration(
    expectation_type="expect_column_distinct_values_to_be_in_set",
    kwargs={
        "column": "change_in_meds_during_hospitalization",
        "value_set" : ['No', 'Ch']
    },
    )
    suite_hospital.add_expectation(expectation_configuration=expectation_change_in_meds_during_hospitalization)

    #prescribed_diabetes_meds
    expectation_prescribed_diabetes_meds = ExpectationConfiguration(
    expectation_type="expect_column_distinct_values_to_be_in_set",
    kwargs={
        "column": "prescribed_diabetes_meds",
        "value_set" : ['Yes', 'No']
    },
    )
    suite_hospital.add_expectation(expectation_configuration=expectation_prescribed_diabetes_meds)

    #outpatient_visits_in_previous_year
    expectation_outpatient_visits_in_previous_year = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "outpatient_visits_in_previous_year",
            "max_value": None,
            "min_value": 0
        },
    )
    suite_hospital.add_expectation(expectation_configuration=expectation_outpatient_visits_in_previous_year)

    #number_lab_tests
    expectation_number_lab_tests = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "number_lab_tests",
            "max_value": None,
            "min_value": 0
        },
    )
    suite_hospital.add_expectation(expectation_configuration=expectation_number_lab_tests)

    #number_of_medications
    expectation_number_of_medications = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "number_of_medications",
            "max_value": None,
            "min_value": 0
        },
    )
    suite_hospital.add_expectation(expectation_configuration=expectation_number_of_medications)

    #number_diagnoses
    expectation_number_diagnoses = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "number_diagnoses",
            "max_value": None,
            "min_value": 0
        },
    )
    suite_hospital.add_expectation(expectation_configuration=expectation_number_diagnoses)

    #average_pulse_bpm
    expectation_average_pulse_bpm = ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "average_pulse_bpm",
            "max_value": 200,
            "min_value": 30
        },
    )
    suite_hospital.add_expectation(expectation_configuration=expectation_average_pulse_bpm)

    context.add_or_update_expectation_suite(expectation_suite=suite_hospital)

    data_asset_name = "test"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe= df)
    except:
        logger.info("The data asset alread exists. The required one will be loaded.")
        data_asset = datasource.get_asset(data_asset_name)

    batch_request = data_asset.build_batch_request(dataframe= df)

    checkpoint = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint_hospital",
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "Hospital",
            },
        ],
    )
    checkpoint_result = checkpoint.run()

    df_validation = get_validation_results(checkpoint_result)
    
    #based on these results you can make an assert to stop your pipeline
    pd_df_ge = gx.from_pandas(df)

    assert pd_df_ge.expect_column_values_to_be_of_type("average_pulse_bpm", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("number_diagnoses", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("number_of_medications", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("number_lab_tests", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("outpatient_visits_in_previous_year", "int64").success == True

    assert pd_df_ge.expect_column_values_to_be_of_type("gender", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("race", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("change_in_meds_during_hospitalization", "str").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("prescribed_diabetes_meds", "str").success == True
    
    # raw dataset file should have 33 columns
    # included 2 columns (index and datetime) after ingestion step
    assert pd_df_ge.expect_table_column_count_to_equal(33).success == True

    log = logging.getLogger(__name__)
    log.info("Data passed on the unit data tests")
  
    return df_validation