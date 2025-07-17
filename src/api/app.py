from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
import logging
from src.menu.manager import Manager

# Configure logging to save all logs to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[logging.FileHandler('app.log', encoding='utf-8')])

logger = logging.getLogger(__name__)

manager = Manager()
app = FastAPI()

# Data model for data and target column
class ModelDataSetup(BaseModel):
    data: list[dict]
    target_column: str

# Data model for user input for prediction 
class UserInput(BaseModel):
    pass

# Endpoint to create and train a model from provided data and target column.
@app.post("/model")
def create_model_from_data(table_data: ModelDataSetup):
    logger.info(f"Received /model request with target_column: {table_data.target_column}")
    try:
        df = pd.DataFrame(table_data.data)
        manager.clean_data(df)
        manager.create_model(table_data.target_column)
        manager.create_model_test()
        manager.create_predictor()
        logger.info("Model setup successful")
        return {"message": "Model setup successful"}
    except Exception as e:
        logger.error(f"Error in /model: {e}")
        return {"Error": str(e)}

#Endpoint to get a prediction for user input data.
@app.post("/predict")
def get_predict(user_data: UserInput):
    logger.info(f"Received /predict request")
    try:
        series = pd.Series(user_data.model_dump())
        predict =  manager.predictor.predict(series)
        return {
                "prediction":predict,
                }
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        return {"Error": str(e)}

#Endpoint to get the model accuracy.
@app.get("/precision")
def get_precision():
    logger.info("Received /precision request")
    try:
        precision = manager.test_model.test_model()
        return {"model precision": precision}
    except Exception as e:
        logger.error(f"Error in /precision: {e}")
        return {"Error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1", port=8000)
