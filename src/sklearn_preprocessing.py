from __future__ import print_function
import os
import json
import argparse
import numpy as np
import pandas as pd
from io import StringIO
from src.config import config as cfg
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from src.pipeline.custom_pipeline import ColumnSelector, ConvertDtypes, GetDummies
from sagemaker_containers.beta.framework import encoders, worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    args = parser.parse_args()

    # Read the data
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(input_files) == 0:
        raise ValueError(f'There are no files in {args.train}. \n'
                         f'This usually indicates that the channel ("train") was incorrectly specified, \n'
                         f'the data specification in S3 was incorrectly specified or the role specified\n'
                         f'does not have permission to acces the data.')
    raw_data = [pd.read_csv(file, names=cfg.FEATURES + [cfg.LABEL],
                            sep=',', dtype=cfg.FEATURES_DTYPES.update(cfg.LABEL_DTYPE)) for file in input_files]
    data = pd.concat(raw_data)
    # Build Pipeline
    preprocessor = Pipeline(steps=[
        ('dtypes', ConvertDtypes(numerical=cfg.NUMERICAL_FEATURES, categorical=cfg.CATEGORICAL_FEATURES)),
        ('selector', ColumnSelector(columns=cfg.FEATURES[3:])),
        ('ohe', GetDummies(columns=cfg.CATEGORICAL_FEATURES))
    ])
    preprocessor.fit(data)
    joblib.dump(preprocessor, filename=os.path.join(args.model_dir, 'preprocessor.joblib'))
    print('The model has been saved!')
    return None


def model_fn(model_dir):
    """

    :param model_dir:
    :return:
    """
    preprocessor_job = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    return preprocessor_job


def input_fn(input_data, content_type):
    """

    :param input_data:
    :param content_type:
    :return:
    """
    if content_type == 'text/csv':
        df = pd.read_csv(StringIO(input_data), sep=',', header=None)
        if len(df.columns) == len(cfg.FEATURES) + 1:
            df.columns = cfg.FEATURES + [cfg.LABEL]
        elif len(df.columns) == len(cfg.FEATURES):
            df.columns = cfg.FEATURES
        return df
    else:
        raise ValueError(f'{content_type} not supported by script')


def predict_fn(input_data, model):
    """

    :param input_data:
    :param model:
    :return:
    """
    features = model.transform(input_data)
    if cfg.LABEL in input_data:
        return np.insert(features, 0, input_data[cfg.LABEL], axis=1)
    else:
        return features


def output_fn(prediction, accept):
    """

    :param prediction:
    :param accept:
    :return:
    """
    if accept == 'application/json':
        instances = []
        for row in prediction.tolist():
            instances.append({'features': row})

        json_output = {'instances': instances}
        return worker.Response(json.dumps(json_output), accept=accept, mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), accept=accept, mimetype=accept)
    else:
        raise RuntimeError(f'{accept} accept type is not supported by this script.')


if __name__ == '__main__':
    main()
