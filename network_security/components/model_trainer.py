import sys,os

import pandas as pd
import numpy as np

from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

from network_security.entity.config_entity import ModelTrainerConfig
from network_security.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact

from network_security.utils.ml_utils.model.estimator import NetworkModel
from network_security.utils.ml_utils.metric.classification_metric import get_classification_score
from network_security.utils.main_utils.utils import save_object,load_object,load_numpy_array_data,evaluate_models

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier

import mlflow
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_experiment("network_security_project")


class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys) from None
    

    def track_mlflow(self, best_model, train_metric, test_metric, X_sample, preprocessor):
        mlflow.set_registry_uri(os.getenv("MLFLOW_TRACKING_URI"))
        tracking_scheme = urlparse(mlflow.get_tracking_uri()).scheme

        wrapper_model = NetworkModel(preprocessor=preprocessor, model=best_model)

        if isinstance(X_sample, pd.DataFrame):
            input_example = X_sample.head(5)
        else:
            input_example = pd.DataFrame(X_sample).head(5)

        try:
            signature = infer_signature(input_example, wrapper_model.predict(input_example))
        except Exception:
            signature = None

        with mlflow.start_run(run_name=f"train_{type(best_model).__name__}"):

            mlflow.set_tag("model_name", type(best_model).__name__)
            mlflow.log_params(best_model.get_params())

            mlflow.log_metric("train_f1_score", train_metric.f1_score)
            mlflow.log_metric("train_precision_score", train_metric.precision_score)
            mlflow.log_metric("train_recall_score", train_metric.recall_score)

            mlflow.log_metric("test_f1_score", test_metric.f1_score)
            mlflow.log_metric("test_precision_score", test_metric.precision_score)
            mlflow.log_metric("test_recall_score", test_metric.recall_score)

            registered_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "network_security_model")

            if tracking_scheme != "file":
                mlflow.sklearn.log_model(
                    sk_model=wrapper_model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_name,
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=wrapper_model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                )

            # ✅ log raw model + preprocessor as separate artifacts
            mlflow.log_artifact("final_model/model.pkl")
            mlflow.log_artifact("final_model/preprocessor.pkl")


    def train_model(self, X_train, y_train, X_test, y_test):

        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=500, solver="lbfgs"),
            "AdaBoost": AdaBoostClassifier()
        }

        params = {
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
        }

        model_report = evaluate_models(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            models=models, param=params
        )

        best_model_name = max(model_report, key=lambda m: model_report[m]["test_f1"])
        best_model = model_report[best_model_name]["best_model"]

        y_train_pred = best_model.predict(X_train)
        train_metric = get_classification_score(y_train, y_train_pred)

        y_test_pred = best_model.predict(X_test)
        test_metric = get_classification_score(y_test, y_test_pred)

        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

        # ✅ Save raw model + preprocessor separately for cloud inference
        os.makedirs("final_model", exist_ok=True)
        save_object("final_model/model.pkl", best_model)
        save_object("final_model/preprocessor.pkl", preprocessor)

        # ✅ Save wrapped model only for local inference/testing
        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, network_model)

        # ✅ Track in Mlflow
        self.track_mlflow(best_model, train_metric, test_metric, pd.DataFrame(X_train[:5]), preprocessor)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric
        )

        logging.info(f"Model trainer artifact : {model_trainer_artifact}")
        return model_trainer_artifact


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            model_trainer_artifact = self.train_model(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test
            )
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys) from None
