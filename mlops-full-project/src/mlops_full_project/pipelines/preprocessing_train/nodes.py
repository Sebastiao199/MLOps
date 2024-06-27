"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""
import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from lib.functions import fill_na, rename_columns, fill_race, medication_cleaning, medication_grouping, age_transformation, fill_age, payer_code_grouping, admission_type_grouping, medical_specialty_grouping, discharge_disposition_grouping, diagnosis_types_grouping, label_pulse


from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import category_encoders as ce



def clean_data(data: pd.DataFrame,) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Does dome data cleaning.
    Args:
        data: Data containing features and target.
    Returns:
        data: Cleaned data
    """
    df_cleaned = data.copy()

    describe_to_dict = df_cleaned.describe().to_dict()

    df_cleaned.replace(['?', ''], value=pd.NA, inplace=True)
    df_cleaned.set_index('encounter_id', inplace=True)

    df_cleaned.drop(['readmitted_multiclass'], axis=1, inplace=True)
    df_cleaned.drop('country', axis=1, inplace=True)
    df_cleaned.drop('weight', axis=1, inplace=True)
    df_cleaned.drop('index', axis=1, inplace=True)
    df_cleaned.drop('datetime', axis=1, inplace=True)

    fill_na(df_cleaned)
    rename_columns(df_cleaned)

    fill_race(df_cleaned)

    medication_cleaning(df_cleaned)
    medication_grouping(df_cleaned)

    age_transformation(df_cleaned)
    fill_age(df_cleaned)

    payer_code_grouping(df_cleaned)

    admission_type_grouping(df_cleaned)
    medical_specialty_grouping(df_cleaned)
    discharge_disposition_grouping(df_cleaned)
    diagnosis_types_grouping(df_cleaned)

    df_cleaned['glucose_test_result'] = df_cleaned['glucose_test_result'].replace({'>200': 'High', '>300': 'High'})
    df_cleaned['a1c_test_result'] = df_cleaned['a1c_test_result'].replace({'>7': 'High', '>8': 'High'})

    df_cleaned['Midpoint_Age'] = df_cleaned['Midpoint_Age'].replace('Unknown', np.nan)

    describe_to_dict_verified = df_cleaned.describe().to_dict()

    return df_cleaned, describe_to_dict_verified 


def feature_engineer(data: pd.DataFrame):
    
    df = data.copy()

    if "readmitted_binary" in df.columns:
        df["readmitted_binary"] = df["readmitted_binary"].map({"No":0, "Yes":1})
    
    #new profiling feature
    # In this step we should start to think on feature store


    # Feature Engineering

    df['race_caucasian'] = df['race'].apply(lambda x: 1 if x == 'Caucasian' else 0)
    df['gender_binary'] = np.where(df['gender']=='Male',1,0)
    df.drop('gender', axis=1, inplace=True)

    df['presc_diabetes_meds_binary'] = np.where(df['prescribed_diabetes_meds']== 'Yes',1,0)
    df.drop('prescribed_diabetes_meds', axis=1, inplace=True)

    df['change_in_meds_binary'] = np.where(df['change_in_meds']=='Ch',1,0)
    df.drop('change_in_meds', axis=1, inplace=True)

    df['Has_Insurance'] = df['payer_code'].apply(lambda x: 0 if x == 'None' else 1)

    df['is_normal_pulse'] = df.apply(lambda row: label_pulse(row), axis=1)

    df['number_encounters_total'] = df.groupby('patient_id')['patient_id'].transform('count')
    df.drop(['patient_id'], axis = 1, inplace = True) 

    df['Total_visits'] = df['inpatient_visits'] + df['outpatient_visits'] + df['emergency_visits']
    df['Serious_condition_visits'] = df['inpatient_visits'] + df['emergency_visits']

    # numerical_features = df.select_dtypes(exclude=['object','string','category']).columns.tolist()
    # categorical_features = df.select_dtypes(include=['object','string','category']).columns.tolist()
    #Exercise create an assert for numerical and categorical features
    
    # OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # OH_cols= pd.DataFrame(OH_encoder.fit_transform(df[categorical_features]))

    # # Adding column names to the encoded data set.
    # OH_cols.columns = OH_encoder.get_feature_names_out(categorical_features)

    # # One-hot encoding removed index; put it back
    # OH_cols.index = df.index

    # # Remove categorical columns (will replace with one-hot encoding)
    # num_df = df.drop(categorical_features, axis=1)

    # # Add one-hot encoded columns to numerical features
    # df_final = pd.concat([num_df, OH_cols], axis=1)


    # Encoding categorical columns using CountEncoder
    
    columns_to_encode = ['race','payer_code','admission_type','medical_specialty','discharge_disposition',
                     'admission_source','primary_diagnosis_types','secondary_diagnosis_types',
                     'additional_diagnosis_types','glucose_test_result','a1c_test_result']

    other_columns = pd.DataFrame(df.drop(columns_to_encode, axis=1))

    count_encoder = ce.CountEncoder(cols=columns_to_encode)

    # fit and transform on your data
    count_encoder.fit(df[columns_to_encode])
    cols_encoded = count_encoder.transform(df[columns_to_encode])

    df_encoded = pd.concat([cols_encoded, other_columns], axis=1)

    # Imputing missing values in 'Midpoint_Age' using a Random Forest Regressor

    known_age = df_encoded[df_encoded['Midpoint_Age'].notnull()]
    unknown_age = df_encoded[df_encoded['Midpoint_Age'].isnull()]

    # Reset indices
    known_age = known_age.reset_index(drop=True)
    unknown_age = unknown_age.reset_index(drop=True)

    # Create the training data for the model
    X = known_age.drop(['Midpoint_Age'], axis=1)
    y = known_age['Midpoint_Age']

    # Fit the model
    regr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
    regr.fit(X, y)

    # Predict the missing ages
    predicted_ages = regr.predict(unknown_age.drop(['Midpoint_Age'], axis=1))

    # Fill in the missing values using the original indices
    df_encoded.loc[df_encoded['Midpoint_Age'].isnull(), 'Midpoint_Age'] = predicted_ages

    log = logging.getLogger(__name__)

    log.info(f"The final dataframe has {len(df_encoded.columns)} columns.")

    return df_encoded, count_encoder, regr


