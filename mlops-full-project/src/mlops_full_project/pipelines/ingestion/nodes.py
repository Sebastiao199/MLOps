"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from great_expectations.core import ExpectationSuite, ExpectationConfiguration


from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)

def build_expectation_suite(expectation_suite_name: str, feature_group: str) -> ExpectationSuite:
    """
    Builder used to retrieve an instance of the validation expectation suite.
    
    Args:
        expectation_suite_name (str): A dictionary with the feature group name and the respective version.
        feature_group (str): Feature group used to construct the expectations.
             
    Returns:
        ExpectationSuite: A dictionary containing all the expectations for this particular feature group.
    """
    
    expectation_suite_hospital = ExpectationSuite(
        expectation_suite_name=expectation_suite_name
    )
    

    # numerical features
    if feature_group == 'numerical_features':

        for i in ['outpatient_visits_in_previous_year', 'emergency_visits_in_previous_year', 
                'inpatient_visits_in_previous_year', 'average_pulse_bpm', 'length_of_stay_in_hospital', 
                'number_lab_tests', 'non_lab_procedures', 'number_of_medications', 'number_diagnoses', 
                'patient_id', 'encounter_id']:    
                    expectation_suite_hospital.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "int64"},
                )
            )

    if feature_group == 'categorical_features':

        # Review engineered features with raw CSV dataset
        expectation_suite_hospital.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "race", "value_set": ['Caucasian', 'AfricanAmerican','Unknown', 'Hispanic', 'Other', 'Asian']},
            )
        )
        expectation_suite_hospital.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "gender", "value_set": ['Female', 'Male']}, # 'Unknown/Invalid'
            )
        )
        expectation_suite_hospital.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "admission_type", "value_set": ['Emergency', 'Elective', 'Newborn', 'Urgent', 'Not Available', 'Not Mapped', 'Trauma Center']},
            ) 
        )
        expectation_suite_hospital.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "change_in_meds_during_hospitalization", "value_set": ['No', 'Ch']},
            )
        )
        expectation_suite_hospital.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "discharge_disposition", "value_set": [
                        'Discharged to home',
                        'Discharged/transferred to a federal health care facility.',
                        'Discharged/transferred to home with home health service',
                        'Discharged/transferred to SNF',
                        'Hospice / medical facility',
                        'Discharged/transferred to another short term hospital',
                        'Discharged/transferred to ICF',
                        'Expired',
                        'Discharged/transferred to another type of inpatient care institution',
                        'Discharged/transferred to another rehab fac including rehab units of a hospital .',
                        'Discharged/transferred to a long term care hospital.',
                        'Left AMA',
                        'Hospice / home',
                        'Discharged/transferred to home under care of Home IV provider',
                        'Not Mapped',
                        'Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital',
                        'Discharged/transferred within this institution to Medicare approved swing bed',
                        'Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.',
                        'Admitted as an inpatient to this hospital',
                        'Discharged/transferred/referred to this institution for outpatient services',
                        'Neonate discharged to another hospital for neonatal aftercare',
                        'Expired at home. Medicaid only, hospice.',
                        'Still patient or expected to return for outpatient services',
                        'Discharged/transferred/referred another institution for outpatient services',
                        'Expired in a medical facility. Medicaid only, hospice.'
                     ]},
            )
        )

        # validate categorical age as a expectation?
        # validate categorical weight as a expectation? There are too much missing values.
        # Which other categories we could use? There are some with too much classes.

    if feature_group == 'target':
        
        expectation_suite_hospital.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "readmitted_binary", "value_set": ['No', 'Yes']},
            )
        ) 
     
    return expectation_suite_hospital


import hopsworks

def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: dict,
    validation_expectation_suite: ExpectationSuite,
    credentials_input: dict
):
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group. 
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
    Returns:
        A dictionary with the feature view version, feature view name and training dataset feature version.
    
    
    """
    # Connect to feature store.
    project = hopsworks.login(
        api_key_value=credentials_input["FS_API_KEY"], project=credentials_input["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    # Create feature group.
    object_feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description= description,
        primary_key=["index"],
        event_time="datetime",
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )
    # Upload data.
    object_feature_group.insert(
        features=data,
        overwrite=False,
        validation_options={"run_validation": False,}, # ignore issues on unclean CVS file
        write_options={
            "wait_for_job": True,
        },
    )

    # Add feature descriptions.

    for description in group_description:
        object_feature_group.update_feature_description(
            description["name"], description["description"]
        )

    # Update statistics.
    object_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    object_feature_group.update_statistics_config()
    object_feature_group.compute_statistics()

    return object_feature_group


def ingestion(
    df: pd.DataFrame, #dataset
    parameters: Dict[str, Any]):

    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group. 
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
    Returns:
       
    
    
    """

    # Possible assertion - review
    #assert len(df.to_)>0, "Wrong data collected"
    
    logger.info(f"The dataset contains {len(df.columns)} columns.")

    numerical_features = df.select_dtypes(exclude=['object','string','category']).columns.tolist()
    #categorical_features = df.select_dtypes(include=['object','string','category']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object','string','category']).columns.tolist()
    categorical_features.remove(parameters["target_column"])

    # Include time column for hopsworks usage
    df['datetime'] = pd.Timestamp('2024-06-25') #datetime.now()
    # Reset index column for hopsworks usage
    df = df.reset_index()
    # Converto all ? marks to NA
    df.replace(['?', '', ' '], value=pd.NA, inplace=True)
    # Fill NAs on relevant features to known classes
    df = fill_na(df)

    # Force conversion of all objects to str
    # df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).astype(str)
    # Get all data types and convert to DataFrame for better formatting
    #dtypes_df = df.dtypes.reset_index()
    #dtypes_df.columns = ['Column', 'Data Type']
    # Print the DataFrame
    #print(dtypes_df)

    validation_expectation_suite_numerical = build_expectation_suite("numerical_expectations","numerical_features")
    validation_expectation_suite_categorical = build_expectation_suite("categorical_expectations","categorical_features")
    validation_expectation_suite_target = build_expectation_suite("target_expectations","target")

    numerical_feature_descriptions = []
    categorical_feature_descriptions = []
    target_feature_descriptions = []
    
    df_numeric = df[["index", "datetime"] + numerical_features] # create dataframes with index column needed by Hopsworks 
    df_categorical = df[["index", "datetime"] + categorical_features]
    df_target = df[["index", "datetime"] + [parameters["target_column"]]] 

    if parameters["to_feature_store"]:

        object_fs_numerical_features = to_feature_store(
            df_numeric,"numerical_features",
            1,"Numerical Features",
            numerical_feature_descriptions,
            validation_expectation_suite_numerical,
            credentials["feature_store"]
        )

        #df_categorical.drop(2115, inplace=True) # problematic row
        #df_categorical.drop(2116, inplace=True) # problematic row
        #df_categorical.drop(2117, inplace=True) # problematic row

        object_fs_categorical_features = to_feature_store(
            df_categorical,"categorical_features", # 2115 - last good row, but breaks in the end, also between 4089-4090
            1,"Categorical Features",
            categorical_feature_descriptions,
            validation_expectation_suite_categorical,
            credentials["feature_store"]
        )

        object_fs_target_features = to_feature_store(
            df_target,"target_features",
           1,"Target Features",
            target_feature_descriptions,
            validation_expectation_suite_target,
            credentials["feature_store"]
        )
    return df

def fill_na(data: pd.DataFrame) -> pd.DataFrame:
    # Check if input is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # Define the columns and their corresponding fill values
    fill_values = {
        'country': 'Unknown',
        'race': 'Unknown',
        'gender': 'Unknown',
        'age': 'Unknown',
        'weight': 'Unknown',
        'payer_code': 'None',
        'admission_type': 'Unknown',
        'medical_specialty': 'Unknown',
        'discharge_disposition': 'Unknown',
        'admission_source': 'Unknown',
        'primary_diagnosis': 'Unknown',
        'secondary_diagnosis': 'Unknown',
        'additional_diagnosis': 'Unknown',
        'glucose_test_result': 'Not_taken',
        'a1c_test_result': 'Not_taken',
        'change_in_meds_during_hospitalization': 'Unknown',
        'prescribed_diabetes_meds': 'Unknown',
        'medication': 'Unknown',
        'readmitted_multiclass': 'Unknown'
    }
    
    # Fill NA values with specified values
    for column, fill_value in fill_values.items():
        if column in data.columns:
            data[column].fillna(fill_value, inplace=True)
    
    return data