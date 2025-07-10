from fastapi import FastAPI
import uvicorn
import pandas as pd
from src.menu.manager import Manager

manager = Manager()
app = FastAPI()


@app.get("/predict")
def get_predict( age: str, income: str, student: str, credit_rating: str):
        data = {'age': age, 'income': income, 'student': student, 'credit_rating': credit_rating}
        series = pd.Series(data=data, index=['age', 'income', 'student', 'credit_rating'])
        return {"prediction": manager.test_model.predict(series)}

@app.get("/precision")
def get_precision():
    return {"model precision": manager.test_model.test_model()}










if __name__ == "__main__":
    uvicorn.run(app,    host="127.0.0.1", port=8000)
