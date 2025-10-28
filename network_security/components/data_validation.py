from network_security.exception import NetworkSecurityException
from network_security.logging.logger import logging

import sys,os

from network_security.entity.config_entity import DataValidationConfig
from network_security.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from network_security.utils.main_utils.utils import read_yaml_file

from network_security.constant.training_pipeline import SCHEMA_FILE_PATH



class DataValidation:
    def __init__(self
        ,data_ingestion_artifact:DataIngestionArtifact
        ,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self.schema_config=read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e,sys)
