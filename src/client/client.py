import pandas as pd
import requests

# Client class for communicating with the FastAPI backend
class Client:
    def __init__(self):
        self.__BASE_URL = "http://127.0.0.1:8000"

    # Send data and target column to backend to create a model
    def create_model(self, data: pd.DataFrame, target: str):
        try:
            data_json = data.to_dict('records')
            response = requests.post(f"{self.__BASE_URL}/model", json={"data": data_json, "target_column": target})
            return response.json()
        except Exception as e:
            return {"Error": e}

    # Send user input to backend to get a prediction
    def get_prediction(self, user_data: dict):
        try:
            response = requests.post(f"{self.__BASE_URL}/predict", json=user_data)
            return response.json()
        except Exception as e:
            return {"Error": e}

    # Get the model accuracy from the backend
    def get_precision(self):
        try:
            response = requests.get(f"{self.__BASE_URL}/precision")
            return response.json()
        except Exception as e:
            return {"Error": e}








