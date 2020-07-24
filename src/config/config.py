import sys
import logging
import numpy as np


# Logger Configuration -----------------------------------------------------------------------------------------------

logger_app_name = 'SageMaker-Exercise'
logger = logging.getLogger(logger_app_name)
logger.setLevel(logging.INFO)
consoleHandle = logging.StreamHandler(sys.stdout)
consoleHandle.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
consoleHandle.setFormatter(formatter)
logger.addHandler(consoleHandle)

# AWS ----------------------------------------------------------------------------------------------------------------

AWS_REGION = 'us-east-1'

# S3

S3_BUCKET = 'banking-data'
S3_PREFIX = 'stiven-lopez'

# Project ------------------------------------------------------------------------------------------------------------


SEED = 42
INPUT = 'Churn_Modelling.csv'
LABEL = 'Exited'
FEATURES = ['RowNumber', 'CustomerId', 'Surname', 'CreditScore',
            'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
NUMERICAL_FEATURES = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
CATEGORICAL_FEATURES = ['Geography', 'Gender']
TO_DROP = ['RowNumber', 'CustomerId', 'Surname']


# Data types

FEATURES_DTYPES = {
    'RowNumber': np.int64,
    'CustomerId': np.int64,
    'Surname': 'category',
    'CreditScore': np.int64,
    'Geography': 'category',
    'Gender': 'category',
    'Age': np.int64,
    'Tenure': np.int64,
    'Balance': np.float64,
    'NumOfProducts': np.int64,
    'HasCrCard': np.int64,
    'IsActiveMember': np.int64,
    'EstimatedSalary': np.float64
}
LABEL_DTYPE = {
    'Exited': np.int64
}

# Categories for features

GEOGRAPHY_CAT = ['France', 'Spain', 'Germany']
GENDER_CAT = ['Female', 'Male']

# Path ---------------------------------------------------------------------------------------------------------------

DATA_FOLDER = 'data'
MODEL_FOLDER = 'models'