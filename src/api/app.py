from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
from src.menu.manager import Manager

manager = Manager()
app = FastAPI()


class ModelDataSetup(BaseModel):
    data: list[dict]
    target_column: str

class UserInput(BaseModel):
    pass

@app.post("/model")
def create_model_from_data(table_data: ModelDataSetup):
    try:
        df = pd.DataFrame(table_data.data)
        manager.clean_data(df)
        manager.create_model(table_data.target_column)
        manager.create_model_test()
        return {"message": "Model setup successful"}
    except Exception as e:
        return {"Error": str(e)}


@app.post("/predict")
def get_predict(user_data: UserInput):
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
    try:
        return {"model precision": manager.test_model.test_model()}
    except Exception as e:
        return {"Error": e}



if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1", port=8000)
