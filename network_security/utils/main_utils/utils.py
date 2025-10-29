import sys,os
import yaml

from network_security.exception import NetworkSecurityException
from network_security.logging.logger import logging

import numpy as np
import dill

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


