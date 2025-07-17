import pandas as pd
import requests
import logging

logger = logging.getLogger(__name__)

# Client class for communicating with the FastAPI backend
class Client:
    def __init__(self):
        self.__BASE_URL = "http://127.0.0.1:8000"

    # Send data and target column to backend to create a model
    def create_model(self, data: pd.DataFrame, target: str):
        logger.info(f"Sending POST to /model")
        try:
            data_json = data.to_dict('records')
            response = requests.post(f"{self.__BASE_URL}/model", json={"data": data_json, "target_column": target})
            logger.info(f"Received response from /model: {response.status_code}")
            return response.json()
        except Exception as e:
            logger.error(f"Error in create_model: {e}")
            return {"Error": str(e)}

    # Send user input to backend to get a prediction
    def get_prediction(self, user_data: dict):
        logger.info(f"Sending POST to /predict")
        try:
            response = requests.post(f"{self.__BASE_URL}/predict", json=user_data)
            logger.info(f"Received response from /predict: {response.status_code}")
            return response.json()
        except Exception as e:
            logger.error(f"Error in get_prediction: {e}")
            return {"Error": str(e)}

    # Get the model accuracy from the backend
    def get_precision(self):
        logger.info("Sending GET to /precision")
        try:
            response = requests.get(f"{self.__BASE_URL}/precision")
            logger.info(f"Received response from /precision: {response.status_code}")
            return response.json()
        except Exception as e:
            logger.error(f"Error in get_precision: {e}")
            return {"Error": str(e)}








