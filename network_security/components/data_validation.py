from network_security.exception import NetworkSecurityException
from network_security.logging.logger import logging

import sys,os

from network_security.entity.config_entity import DataValidationConfig
from network_security.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from network_security.utils.main_utils.utils import read_yaml_file,write_yaml_file

from network_security.constant.training_pipeline import SCHEMA_FILE_PATH

from scipy.stats import ks_2samp
import pandas as pd




class DataValidation:
    def __init__(self
        ,data_ingestion_artifact:DataIngestionArtifact
        ,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e,sys) from None
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys) from None
    
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=len(self._schema_config['columns'])-1
            logging.info(f"Schema expects {number_of_columns} columns. Dataframe has {len(dataframe.columns)} columns.")
            return len(dataframe.columns)==number_of_columns
        except Exception as e:
            raise NetworkSecurityException(e,sys) from None
    
    def validate_number_of_numerical_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_numerical_columns=len(self._schema_config['columns']['numerical_columns'])
            numerical_columns_in_dataframe=dataframe.select_dtypes(include='number')
            logging.info(f"Required number of columns : {number_of_numerical_columns}")
            logging.info(f"Data frame has columns : {len(numerical_columns_in_dataframe)}")
            return len(numerical_columns_in_dataframe)==number_of_numerical_columns
        except Exception as e:
            raise NetworkSecurityException(e,sys) from None

    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        status=True
        try:
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                is_drift_detected = is_same_dist.pvalue < threshold
                if is_drift_detected:
                    status = False
                report[column] = {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_detected": is_drift_detected
                }            
            ### saving the drift report
            drift_report_file_path=self.data_validation_config.drift_report_file_path
            dir_path=os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)
            
        except Exception as e:
            raise NetworkSecurityException(e,sys) from None
        
        return status


    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.trained_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path
            
            ### read the data from the artifact
            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)

            ### validate number of columns
            error_message=""
            status=True
            status=self.validate_number_of_columns(dataframe=train_dataframe) and status
            if not status:
                error_message=f"{error_message} Train dataframe does not contain all the columns\n"

            status=self.validate_number_of_columns(dataframe=test_dataframe) and status
            if not status:
                error_message=f"{error_message} Test dataframe does not contain all the columns\n"

            ### validate numerical columns
            status=self.validate_number_of_numerical_columns(dataframe=train_dataframe) and status
            if not status:
                error_message=f"{error_message} Train dataframe does not contain all the numerical columns\n"

            status=self.validate_number_of_numerical_columns(dataframe=test_dataframe) and status
            if not status:
                error_message=f"{error_message} Test dataframe does not contain all the numerical columns\n"
            if error_message!="":
                logging.error(error_message)
            
            # check data drift
            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe) and status
            os.makedirs(os.path.dirname(self.data_validation_config.validated_train_file_path),exist_ok=True)
            
            train_dataframe.to_csv(
                    self.data_validation_config.validated_train_file_path,
                    index=False, header=True
                )

            test_dataframe.to_csv(
                self.data_validation_config.validated_test_file_path,
                index=False, header=True
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                validated_train_file_path=self.data_validation_config.validated_train_file_path,
                validated_test_file_path=self.data_validation_config.validated_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys) from None
