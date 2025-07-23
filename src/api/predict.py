from fastapi import FastAPI, HTTPException
import requests
import pickle
import uvicorn
from pydantic import BaseModel, Field
import pandas as pd
import logging
from src.models.predictor import Predictor
from src.core.config import settings
from src.core.logging import setup_logging


setup_logging()
logger = logging.getLogger(__name__)
model = None
predictor = None


# Lifespan event to load the model at startup
async def lifespan(app: FastAPI):
    logger.info("Starting lifespan")
    try:
        global model
        global predictor
        response = requests.get(settings.MODEL_SERVER_URL)
        if response.status_code == 200:
            model = pickle.loads(response.content)
            predictor = Predictor(model)
            logger.info("Model loaded successfully")                
        else:
            logger.error("Failed to load model")
    except Exception as e:  
        logger.error(f"Error during lifespan: {e}")
    yield
    logger.info("Ending lifespan")

    
app = FastAPI(lifespan=lifespan)

# Model input schema
class UserInput(BaseModel):
    Age:int = Field(gt=0, lt=120)
    Gender:str
    IncomeLevel: str
    MaritalStatus: str
    PreviousPurchases: int = Field(gt=-1)
    VisitedWebsite: str

    class Config:
        extra = "allow"

#Endpoint to get a prediction for user input data.
@app.post("/predict")
def get_predict(user_data: UserInput):
    logger.info(f"Received /predict request")
    if not predictor:
        logger.error("Model is not initialized")
        return {"Error": "Model is not initialized"}
    try:
        series = pd.Series(user_data.model_dump())
        predict =  predictor.predict(series)
        return {"prediction": predict}
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        return HTTPException(status_code=500, detail=str(e))

