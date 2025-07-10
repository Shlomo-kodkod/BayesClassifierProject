from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
from src.menu.manager import Manager

manager = Manager()
app = FastAPI()

class ParmsData(BaseModel):
    Index: int
    UsingIP: int
    LongURL: int
    ShortURL: int
    Symbol: int
    Redirecting: int
    PrefixSuffix: int
    SubDomains: int
    HTTPS: int
    DomainRegLen: int
    Favicon: int
    NonStdPort: int
    HTTPSDomainURL: int
    RequestURL: int
    AnchorURL: int
    LinksInScriptTags: int
    ServerFormHandler: int
    InfoEmail: int
    AbnormalURL: int
    WebsiteForwarding: int
    StatusBarCust: int
    DisableRightClick: int
    UsingPopupWindow: int
    IframeRedirection: int
    AgeofDomain: int
    DNSRecording: int
    WebsiteTraffic: int
    PageRank: int
    GoogleIndex: int
    LinksPointingToPage: int
    StatsReport: int



@app.post("/predict")
def get_predict(user_data: ParmsData):
    try:
        series = pd.Series(user_data.model_dump())
        predict =  manager.test_model.predict(series)
        return {
                "prediction":predict,
                "result map": "1 = Phishing Website, -1 = Non-Phishing Website"
                }
    except Exception as e:
        return {"Error": e}



@app.get("/precision")
def get_precision():
    return {"model precision": manager.test_model.test_model()}



if __name__ == "__main__":
    uvicorn.run(app,    host="127.0.0.1", port=8000)
