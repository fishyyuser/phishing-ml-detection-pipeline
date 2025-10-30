import sys

from network_security.exception import NetworkSecurityException
from network_security.logging.logger import logging

from network_security.components.data_ingestion import DataIngestion
from network_security.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig

from network_security.components.data_validation import DataValidation
from network_security.entity.config_entity import DataValidationConfig

from network_security.components.data_transformation import DataTransformation
from network_security.entity.config_entity import DataTransformationConfig

from network_security.components.model_trainer import ModelTrainer
from network_security.entity.config_entity import ModelTrainerConfig



if __name__=="__main__":
    try:
        training_pipeline_config=TrainingPipelineConfig()
        
        logging.info("Initiated the Data Ingestion")
        data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config)
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        print("=="*45)
        print(data_ingestion_artifact)


        logging.info("Initiated the Data Validation")
        data_validation_config=DataValidationConfig(training_pipeline_config)
        data_validation=DataValidation(data_ingestion_artifact,data_validation_config)
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")

        print("=="*45)
        print(data_validation_artifact)


        logging.info("Initiated the Data Transformation")
        data_transformation_config=DataTransformationConfig(training_pipeline_config)
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        logging.info("Data Transformation Completed")

        print("=="*45)
        print(data_transformation_artifact)


        logging.info("Initiated the Model Trainer")
        model_trainer_config=ModelTrainerConfig(training_pipeline_config)
        model_trainer=ModelTrainer(data_transformation_artifact=data_transformation_artifact,model_trainer_config=model_trainer_config)
        model_trainer_artifact=model_trainer.initiate_model_trainer()
        logging.info("Model Trainer Completed")

        print("=="*45)
        print(model_trainer_artifact)

    except Exception as e:
        raise NetworkSecurityException(e,sys) from None
