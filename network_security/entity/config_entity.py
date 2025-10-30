from datetime import datetime
import os
from network_security.constant import training_pipeline

print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=training_pipeline.PIPELINE_NAME
        self.artifact_name=training_pipeline.ARTIFACT_DIR
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        self.timestamp:str=timestamp

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig()):
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )

        self.feature_store_file_path:str=os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,training_pipeline.FILE_NAME
        )

        # both training set and testing set is saved on same folder
        self.training_file_path:str=os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )

        self.testing_file_path:str=os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )

        self.train_test_split_ratio:float=training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name:str=training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name:str=training_pipeline.DATA_INGESTION_DATABASE_NAME

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig()):
        self.data_validation_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )

        self.valid_data_dir:str=os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALIDATED_DIR
        )
        

        self.validated_train_file_path:str=os.path.join(
            self.valid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )
        
        self.validated_test_file_path:str=os.path.join(
            self.valid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )
        self.drift_report_file_path:str=os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )

class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig()):

        self.data_transformation_dir:str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        self.transformed_train_file_path:str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.DATA_TRANSFORMATION_TRAIN_FILE_PATH
        )

        self.transformed_test_file_path:str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.DATA_TRANSFORMATION_TEST_FILE_PATH
        )

        self.transformed_object_file_path:str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
        )
