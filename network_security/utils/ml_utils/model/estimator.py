import sys,os

from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

from network_security.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME

class NetworkModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e,sys) from None
    
    def predict(self,x):
        try:
            x_transformed=self.preprocessor.transform(x)
            y_pred=self.model.predict(x_transformed)
            return y_pred
        except Exception as e:
            raise NetworkSecurityException(e,sys) from None