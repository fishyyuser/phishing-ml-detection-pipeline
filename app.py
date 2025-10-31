import sys
import os

import certifi
ca=certifi.where()

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

from pymongo.mongo_client import MongoClient

from network_security.exception import NetworkSecurityException
from network_security.logging.logger import logging

from network_security.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File,UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse

from fastapi.templating import Jinja2Templates
templates=Jinja2Templates(directory="./templates")

import pandas as pd

client=MongoClient(MONGO_DB_URL,tlsCAFile=ca)

from network_security.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME,DATA_INGESTION_COLLECTION_NAME
from network_security.utils.main_utils.utils import load_object
from network_security.utils.ml_utils.model.estimator import NetworkModel

database=client[DATA_INGESTION_DATABASE_NAME]
collection=database[DATA_INGESTION_COLLECTION_NAME]

app= FastAPI()
origins= ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/",tags={"authentication"})
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e,sys) from None

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # Load preprocessor and model
        preprocessor_path =os.path.join("final_model","preprocessor.pkl")
        model_path = os.path.join("final_model", "model.pkl")
        preprocesor=load_object(preprocessor_path)
        final_model=load_object(model_path)
        network_model = NetworkModel(preprocessor=preprocesor,model=final_model)

        y_pred = network_model.predict(df)

        df['predicted_column'] = y_pred
        df['predicted_column'].replace(-1, 0, inplace=True)

        os.makedirs("prediction_output", exist_ok=True)
        output_path = os.path.join("prediction_output", "output.csv")
        df.to_csv(output_path, index=False)

        table_html = df.to_html(classes='table table-striped')

        return templates.TemplateResponse(
            "table.html",
            {"request": request, "table": table_html}
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__=="__main__":
    app_run(app, host="0.0.0.0", port=8000)
