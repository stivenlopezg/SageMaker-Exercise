import logging
from src.config import config as cfg
from aws.aws_helper import AwsHelper
from utilities.data_wrangling import load_data, split_dataset, concatenate_data, export_data

logger = logging.getLogger(cfg.logger_app_name)

aws = AwsHelper()


def main():
    """
    Execute split_data process
    :return: None
    """
    logger.info('El proceso ha comenzado... \n')
    aws.download_from_s3(bucket=cfg.S3_BUCKET, key=f'{cfg.INPUT}',
                         local_path=f'{cfg.DATA_FOLDER}/{cfg.INPUT}')
    df = load_data(filename=f'{cfg.DATA_FOLDER}/{cfg.INPUT}', sep=',',
                   dtype=cfg.FEATURES_DTYPES.update(cfg.LABEL_DTYPE))
    train_data, validation_data, test_data, train_label, validation_label, test_label = split_dataset(df=df,
                                                                                                      label=cfg.LABEL,
                                                                                                      stratify=False)
    train_df = concatenate_data(train_data, train_label)
    validation_df = concatenate_data(validation_data, validation_label)
    export_data(df=train_df, file_path=f'{cfg.DATA_FOLDER}/train.csv')
    export_data(df=validation_df, file_path=f'{cfg.DATA_FOLDER}/validation.csv')
    export_data(df=test_data, file_path=f'{cfg.DATA_FOLDER}/test.csv')
    export_data(df=test_label, file_path=f'{cfg.DATA_FOLDER}/test_label.csv')
    export_data(df=train_df, file_path=f'{cfg.DATA_FOLDER}/train_with_header.csv', with_header=True)
    aws.upload_to_s3(local_path=f'{cfg.DATA_FOLDER}/train.csv',
                     bucket=cfg.S3_BUCKET, key=f'{cfg.S3_PREFIX}/train/train.csv')
    aws.upload_to_s3(local_path=f'{cfg.DATA_FOLDER}/test.csv',
                     bucket=cfg.S3_BUCKET, key=f'{cfg.S3_PREFIX}/test/test.csv')
    aws.upload_to_s3(local_path=f'{cfg.DATA_FOLDER}/test_label.csv',
                     bucket=cfg.S3_BUCKET, key=f'{cfg.S3_PREFIX}/test/test_label.csv')
    aws.upload_to_s3(local_path=f'{cfg.DATA_FOLDER}/validation.csv',
                     bucket=cfg.S3_BUCKET, key=f'{cfg.S3_PREFIX}/validation/validation.csv')
    aws.upload_to_s3(local_path=f'{cfg.DATA_FOLDER}/train_with_header.csv',
                     bucket=cfg.S3_BUCKET, key=f'{cfg.S3_PREFIX}/baseline/data/train_with_header.csv')
    logger.info('El proceso ha finalizado correctamente.')
    return None


if __name__ == '__main__':
    main()
