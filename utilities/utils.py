import os
import logging
import xgboost
import pandas as pd
from src.config import config as cfg

logger = logging.getLogger(cfg.logger_app_name)


def delete_file(file_path: str):
    """

    :param file_path:
    :return: None
    """
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info('Se ha eliminado el archivo correctamente.')
    else:
        logger.info('No se ha encontrado el archivo')
    return None


def download_model(model_path: str, local_dir: str):
    """

    :param model_path:
    :param local_dir:
    :return:
    """
    return os.system(command=f'aws s3 cp {model_path} {local_dir}')


def decompress_model(local_dir: str):
    """

    :param local_dir:
    :return:
    """
    return os.system(command=f'tar xvf {local_dir}')


def prediction_df(model, file_path: str, score: float):
    """

    :param model:
    :param file_path:
    :param score:
    :return:
    """
    df = pd.read_csv(file_path, sep=',', header=None)
    df.columns = [f'f{i}' for i in range(0, df.shape[1])]
    df['prediction'] = model.predict(xgboost.DMatrix(data=df))
    df['prediction'] = df['prediction'].apply(lambda x: 1 if x >= score else 0)
    return df
