import os
import sys
import json

from network_security.exception import NetworkSecurityException
from network_security.logging.logger import logging

from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL=os.getenv("MONGO_DB_URL")

if not MONGO_DB_URL:
    raise NetworkSecurityException("MONGO_DB_URL not found in environment", sys)

import certifi
ca=certifi.where()

import pandas as pd
import numpy as np

from pymongo import MongoClient

class NetworkDataExtract():
    def __init__(self):
        pass
        
    
    def csv_to_json(self, file_path):
        try:
            logging.info(f"Reading CSV file from: {file_path}")
            data = pd.read_csv(file_path, index_col=False)
            logging.info(f"CSV file read successfully with {data.shape[0]} rows and {data.shape[1]} columns")
            records = json.loads(data.to_json(orient="records"))
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def insert_data_mongoDB(self, records, database, collection):
        try:
            with MongoClient(MONGO_DB_URL, tlsCAFile=ca) as client:
                db = client[database]
                col = db[collection]
                result = col.insert_many(records)
                logging.info(f"Inserted {len(result.inserted_ids)} records into {database}.{collection}")
                return len(result.inserted_ids)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

if __name__=="__main__":
    FILE_PATH = os.path.join("network_data", "phisingData.csv")
    DATABASE="pradyumn"
    COLLECTION="Network_Data"
    networkObj=NetworkDataExtract()
    RECORDS=networkObj.csv_to_json(FILE_PATH)
    no_of_Records=networkObj.insert_data_mongoDB(RECORDS,DATABASE,COLLECTION)
    print(no_of_Records)






