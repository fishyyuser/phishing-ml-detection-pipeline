import sys,os
import yaml

from network_security.exception import NetworkSecurityException
from network_security.logging.logger import logging

import numpy as np
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,'rb')as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from None

def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name != '':
            os.makedirs(dir_name, exist_ok=True)
        
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        
        with open(file_path, 'w') as file:
            yaml.dump(content, file)

        logging.info(f"YAML file saved successfully at: {os.path.abspath(file_path)}")

    except Exception as e:
        raise NetworkSecurityException(e,sys) from None

def save_object(file_path:str,obj:object)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb")as file_obj:
            dill.dump(obj,file_obj)
        logging.info(f"Saved {obj} in the path : {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e,sys) from None

def save_numpy_array_data(file_path:str,array:np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
        logging.info(f"Saved the Numpy array object in the file_path : {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e,sys) from None

def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            logging.error(f"File does not exist at path: {file_path}")
            raise Exception(f"File does not exist at path: {file_path}")

        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
            logging.info(f"Successfully loaded object from: {file_path}")
            return obj

    except Exception as e:
        raise NetworkSecurityException(e, sys) from None


def load_numpy_array_data(file_path: str) -> np.array:
    try:
        if not os.path.exists(file_path):
            logging.error(f"Numpy array file does not exist at path: {file_path}")
            raise Exception(f"Numpy array file does not exist at path: {file_path}")

        with open(file_path, 'rb') as file_obj:
            array = np.load(file_obj)
            logging.info(f"Successfully loaded numpy array from: {file_path}")
            return array

    except Exception as e:
        raise NetworkSecurityException(e, sys) from None

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            params = param[model_name]

            logging.info(f"Running GridSearchCV for {model_name}...")
            gs = GridSearchCV(model, params, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            logging.info(
                f"{model_name}: train_f1={train_f1:.3f}, test_f1={test_f1:.3f}, best_params={gs.best_params_}"
            )

            report[model_name] = {
                "train_f1": train_f1,
                "test_f1": test_f1,
                "best_model":best_model
            }

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys) from None
