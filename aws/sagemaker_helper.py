import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.amazon.amazon_estimator import get_image_uri


def sklearn_preprocessor(entry_point: str, role: str, output_dir: str, instance_type: str = 'ml.m4.xlarge'):
    """

    :param output_dir:
    :param entry_point:
    :param role:
    :param instance_type:
    :return:
    """
    preprocessor = SKLearn(entry_point=entry_point,
                           role=role,
                           source_dir='src',
                           train_instance_type=instance_type,
                           train_use_spot_instances=True,
                           train_max_run=300,
                           train_max_wait=600,
                           output_path=output_dir,
                           dependencies=['src/config/config.py', 'src/pipeline/custom_pipeline.py'])
    return preprocessor


def sagemaker_model(image: str, hyperparams: dict, role: str, output_dir: str,
                    region_name: str = 'us-east-1', instance_type: str = 'ml.m4.xlarge'):
    """

    :param output_dir:
    :param image:
    :param hyperparams:
    :param role:
    :param instance_type:
    :param region_name:
    :return:
    """
    if image == 'xgboost':
        input_mode = 'File'
        container = get_image_uri(region_name, image, '0.90-2')
    else:
        input_mode = 'Pipe'
        container = get_image_uri(region_name, image)
    model = sagemaker.estimator.Estimator(container,
                                          role=role,
                                          input_mode=input_mode,
                                          train_instance_count=1,
                                          output_path=output_dir,
                                          train_instance_type=instance_type,
                                          train_use_spot_instances=True,
                                          train_max_run=300,
                                          train_max_wait=600)
    model.set_hyperparameters(**hyperparams)
    return model


def batch_transform(model, data: str, output_dir: str, instance_type: str = 'ml.m4.xlarge'):
    """

    :param output_dir:
    :param model:
    :param data:
    :param instance_type:
    :return:
    """
    transformer = model.transformer(instance_count=1,
                                    instance_type=instance_type,
                                    max_payload=100,
                                    output_path=output_dir,
                                    assemble_with='Line')
    transformer.transform(data)
    transformer.wait()
    return None
