import sys,os

import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from network_security.exception import NetworkSecurityException
from network_security.logging.logger import logging

from network_security.entity.config_entity import DataTransformationConfig
from network_security.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact

from network_security.utils.main_utils.utils import save_object,save_numpy_array_data
from network_security.constant.training_pipeline import TARGET_COLUMN,DATA_TRANSFORMATION_IMPUTER_PARAMS

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e,sys) from None
    
    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            if not os.path.exists(file_path):
                logging.error(f"File does not exist in the given path {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys) from None

    def get_data_tranformer_object(cls)->Pipeline:
        """
        It initialises a KNNImputer object with the parameters specified in the training_pipeline.py file
        and returns a Pipeline object with the KNNImputer object as the first step.

        Args:
          cls: DataTransformation

        Returns:
          A Pipeline object
        """
        try:
            imputer:KNNImputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Initialised the KNN Imputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            processor=Pipeline([("imputer",imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e,sys) from None

    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Initiated Data Transformation")
        try:
            logging.info("Reading the train and test data from data validation artifacts")
            train_df=DataTransformation.read_data(self.data_validation_artifact.validated_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.validated_test_file_path)

            # training dataframe
            X_train=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            y_train=train_df[TARGET_COLUMN]
            y_train=y_train.replace(-1,0)
            
            # testing dataframe
            X_test=test_df.drop(columns=[TARGET_COLUMN],axis=1)
            y_test=test_df[TARGET_COLUMN]
            y_test=y_test.replace(-1,0)

            preprocessor=self.get_data_tranformer_object()
            
            logging.info("transforming the data using preprocessor")
            preprocessor_obj=preprocessor.fit(X_train)
            X_train_transformed=preprocessor_obj.transform(X_train)
            X_test_transformed=preprocessor_obj.transform(X_test)

            train_arr=np.c_[X_train_transformed,np.array(y_train)]
            test_arr=np.c_[X_test_transformed,np.array(y_test)]

            # save numpy array data
            logging.info("saving the numpy arrays and tranformer object as data transformation artifacts")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_obj)

            preprocessor_file_path=os.path.join("final_model","preprocessor.pkl")
            save_object(preprocessor_file_path,preprocessor_obj)

            # preparing artifacts
            logging.info("Exporting data transformation artifact")
            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys) from None


