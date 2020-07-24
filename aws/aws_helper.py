import boto3
import logging
from src.config import config as cfg
from botocore.exceptions import ClientError

logger = logging.getLogger(cfg.logger_app_name)


class AwsHelper:
    """
    Helper con los metodos mas usados de boto3
    """
    def __init__(self):
        self.s3_client = boto3.client('s3', region_name=cfg.AWS_REGION)

    def download_from_s3(self, bucket: str, key: str, local_path: str):
        """
        Download from S3 a file in specific bucket to a specific local path
        :param bucket: s3 bucket name
        :param key: filename (subfolder/filename)
        :param local_path: local path
        :return: None
        """
        logger.info('Se ha empezado a descargar el objeto ... \n')
        try:
            self.s3_client.download_file(bucket, key, local_path)
            logger.info('Se ha descargado el archivo correctamente.')
        except (Exception, ClientError) as e:
            logger.error(f'Error descargando desde S3, {e}')
            if e.response['Error']['Code'] == '404':
                logging.info('El objeto no existe')
        return None

    def upload_to_s3(self, local_path: str, bucket: str, key: str, with_kms: bool = False):
        """
        Load a file from a specific route to an S3 bucket
        :param local_path: file
        :param bucket: s3 bucket name
        :param key: (subfolder/filename)
        :param with_kms: Boolean if the file is encrypted with kms
        :return: None
        """
        logger.info('Se ha empezado a cargar el archivo a S3 ... \n')
        if with_kms:
            try:
                self.s3_client.upload_file(local_path, bucket, key, extra_args={'ServiceSideEncryption': 'aws:kms',
                                                                                'SS#KMSKeyId': '<<your_kms_key>>'})
                logger.info('Se ha cargado el archivo en S3 correctamente.')
            except (Exception, ClientError) as e:
                logger.error(f'Error caragando a S3, {e}')
        else:
            try:
                self.s3_client.upload_file(local_path, bucket, key)
                logger.info('Se ha cargado el archivo en S3 correctamente.')
            except (Exception, ClientError) as e:
                logging.error(f'Error cargando a S3, {e}')
        return None
