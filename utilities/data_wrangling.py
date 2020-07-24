import logging
import pandas as pd
from src.config import config as cfg
from sklearn.model_selection import train_test_split

logger = logging.getLogger(cfg.logger_app_name)


def load_data(filename: str, **kwargs) -> pd.DataFrame:
    """
    Upload a csv file
    :param filename: file
    :param kwargs: additional parameters
    :return: pd.DataFrame
    """
    df = pd.read_csv(filename, **kwargs)
    logger.info('Se han cargado los datos correctamente.')
    return df


def split_dataset(df: pd.DataFrame, label: str, stratify: bool = False):
    """
    Partition the data in training, testing and validation
    :param df: DataFrame
    :param label: target variable
    :param stratify: If partitioning using stratified sampling
    :return: pd.DataFrame
    """
    target = df.pop(label)
    if stratify:
        train_data, test_data, train_label, test_label = train_test_split(df, target, test_size=0.3,
                                                                          random_state=cfg.SEED, stratify=target)
        test_data, validation_data, test_label, validation_label = train_test_split(test_data, test_label,
                                                                                    train_size=0.4,
                                                                                    random_state=cfg.SEED,
                                                                                    stratify=test_label)
    else:
        train_data, test_data, train_label, test_label = train_test_split(df, target, test_size=0.3,
                                                                          random_state=cfg.SEED)
        test_data, validation_data, test_label, validation_label = train_test_split(test_data, test_label,
                                                                                    train_size=0.4,
                                                                                    random_state=cfg.SEED)
    logger.info('Se ha partido el conjunto de datos en conjuntos de entrenamiento, validacion, y prueba.')
    return train_data, validation_data, test_data, train_label, validation_label, test_label


def concatenate_data(first_df: pd.DataFrame, second_df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins two dataframe
    :param first_df: DataFrame
    :param second_df: DataFrame
    :return: pd.DataFrame
    """
    df = pd.concat([first_df, second_df], axis=1).reset_index(drop=True)
    logger.info('Se han unido ambos set de datos')
    return df


def export_data(df: pd.DataFrame, file_path: str, with_header: bool = False):
    """
    Export the dataframe to a csv file
    :param df: DataFrame
    :param file_path: path that will host the file
    :param with_header: whether the file is exported with header or not
    :return:
    """
    logger.info('El archivo se ha exportado como un csv satisfactoriamente')
    if with_header:
        return df.to_csv(file_path, sep=';', index=False, header=True)
    else:
        return df.to_csv(file_path, sep=';', index=False, header=False)
