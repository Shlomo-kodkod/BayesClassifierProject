from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
from src.menu.manager import Manager

manager = Manager()
app = FastAPI()


class ModelSetup(BaseModel):
    csv_path: str
    target_column: str


@app.post("/model")
def create_model(table_data: ModelSetup):
    try:
        path, target = table_data.csv_path, table_data.target_column
        manager.load_data(path)
        manager.create_model(target)
        manager.create_model_test()
        return {
            "message": "Model setup successful"}
    except Exception as e:
        return {"Error": e}


@app.post("/predict")
def get_predict(user_data):
    try:
        series = pd.Series(user_data.model_dump())
        predict =  manager.test_model.predict(series)
        return {
                "prediction":predict,
                }
    except Exception as e:
        return {"Error": e}

@app.get("/precision")
def get_precision():
    return {"model precision": manager.test_model.test_model()}



if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1", port=8000)
