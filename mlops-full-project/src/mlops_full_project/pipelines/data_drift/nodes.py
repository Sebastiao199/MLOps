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

from .utils import calculate_psi
import matplotlib.pyplot as plt  
import nannyml as nml
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


logger = logging.getLogger(__name__)

def generate_covariate_shift(data, feature, shift_factor=1.5):
    drifted_data = data.copy()
    drifted_data[feature] = drifted_data[feature] * shift_factor
    return drifted_data

def generate_label_shift(data, target, shift_probability=0.2):
    drifted_data = data.copy()
    mask = np.random.random(len(drifted_data)) < shift_probability
    drifted_data.loc[mask, target] = 1 - drifted_data.loc[mask, target]
    return drifted_data

def generate_concept_drift(data, feature, target, drift_factor=0.5):
    drifted_data = data.copy()
    drifted_data[target] = np.where(
        drifted_data[feature] > drifted_data[feature].median(),
        1 - drifted_data[target],
        drifted_data[target]
    )
    return drifted_data




def data_drift(data_reference: pd.DataFrame, data_analysis: pd.DataFrame):
    

    #define the threshold for the test as parameters in the parameters catalog
    constant_threshold = nml.thresholds.ConstantThreshold(lower=None, upper=0.2)
    constant_threshold.thresholds(data_reference)

    # Should we use SHAP output as the most important features to track?
    # Let's initialize the object that will perform the Univariate Drift calculations
    univariate_calculator = nml.UnivariateDriftCalculator(
    column_names=["age"],
    treat_as_categorical=['age'],
    chunk_size=50,
    categorical_methods=['jensen_shannon'],
    thresholds={"jensen_shannon":constant_threshold})

    univariate_calculator.fit(data_reference)
    results = univariate_calculator.calculate(data_analysis).filter(period='analysis', column_names=['age'],methods=['jensen_shannon']).to_df()

    figure = univariate_calculator.calculate(data_analysis).filter(period='analysis', column_names=['age'],methods=['jensen_shannon']).plot(kind='drift')
    figure.write_html("data/08_reporting/univariate_nml.html")
    
    #generate a report for some numeric features using KS test and evidently ai
    data_drift_report = Report(metrics=[
    DataDriftPreset(cat_stattest='ks', stattest_threshold=0.05)])

    data_drift_report.run(current_data=data_analysis[["inpatient_visits_in_previous_year","emergency_visits_in_previous_year"]] , reference_data=data_reference[["inpatient_visits_in_previous_year","emergency_visits_in_previous_year"]], column_mapping=None)
    data_drift_report.save_html("data/08_reporting/data_drift_report.html")
    
    # Should we run calculate_psi() as alternate method? 

    # We tried to drifting categorical data but we were having issues.

    # # Combine multiple types of drift
    # drifted_data = data_analysis.copy()
    # drifted_data['readmitted_binary'] = drifted_data['readmitted_binary'].map({0: 'No', 1: 'Yes'})
    # drifted_data_cov = generate_covariate_shift(drifted_data, 'inpatient_visits_in_previous_year', 1.5)
    # drifted_data_lab = generate_label_shift(drifted_data, 'readmitted_binary', 0.1)

    # # Apply drift to a subset of the data
    # subset_mask = np.random.random(len(data_analysis)) < 0.5
    # drifted_data_con = data_analysis.copy()
    # drifted_data_con['readmitted_binary'] = drifted_data_con['readmitted_binary'].map({0: 'No', 1: 'Yes'})
    # drifted_data_con.loc[subset_mask] = generate_concept_drift(drifted_data_con.loc[subset_mask], 'inpatient_visits_in_previous_year', 'readmitted_binary', 0.3)

    # drifted_data_cov = drifted_data_cov.to_numpy()
    # drifted_data_lab = drifted_data_lab.to_numpy()
    # drifted_data_con = drifted_data_con.to_numpy()

    # data_analysis = data_analysis.to_numpy()
    
    # cov_psi = calculate_psi(data_analysis, drifted_data_cov, buckettype='quantiles')
    # lab_psi = calculate_psi(data_analysis, drifted_data_lab, buckettype='quantiles')
    # con_psi = calculate_psi(data_analysis, drifted_data_con, buckettype='quantiles')

    # logger.info('PSI - Covariate drift:', cov_psi)
    # logger.info('PSI - Label drift:', lab_psi)
    # logger.info('PSI - Concept drift:', con_psi)


    return results