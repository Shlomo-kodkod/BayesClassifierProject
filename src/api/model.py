from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import pickle
import uvicorn
import logging
from src.services.manager import Manager
from src.core.config import settings
from src.core.logging import setup_logging


setup_logging()
logger = logging.getLogger(__name__)
manager = Manager()


# Lifespan event to load the model at startup
async def lifespan(app: FastAPI):
    logger.info("Starting lifespan")
    try:
        manager.create_model("data/customer_purchase_data.csv", "WillBuy")
        manager.create_model_test()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error during lifespan: {e}")
    
    yield
    logger.info("Ending lifespan")


app = FastAPI(lifespan=lifespan)


# Endpoint to get a prediction for user input data.
@app.get("/model")
def get_model():
    logger.info("Received /model request")
    try:
        model_data = pickle.dumps(manager.model_data)
        return Response(content=model_data, media_type="application/octet-stream")
    except Exception as e:
        logger.error(f"Error in /model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#Endpoint to get the model accuracy.
@app.get("/precision")
def get_precision():
    logger.info("Received /precision request")
    try:
        precision = manager.test_model.test_model()
        return {"model precision": precision}
    except Exception as e:
        logger.error(f"Error in /precision: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#Endpoint to get the confusion matrix.
@app.get("/matrix")
def confusion_matrix():
    logger.info("Received /matrix request")
    try:
        matrix = manager.test_model.confusion_matrix()
        logger.info("Confusion matrix retrieved successfully")
        return {"confusion matrix": matrix}
    except Exception as e:
        logger.error(f"Error in /: {e}")
        raise HTTPException(status_code=500, detail=str(e))
