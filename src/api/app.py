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
    handlers=[logging.FileHandler('server.log', encoding='utf-8')])

logger = logging.getLogger(__name__)

manager = Manager()
manager.create_model("data/customer_purchase_data.csv", "WillBuy")
manager.create_predictor()
manager.create_model_test()
app = FastAPI()


class UserInput(BaseModel):
    pass

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
