from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
from src.menu.manager import Manager

manager = Manager()
app = FastAPI()

class ParmsData(BaseModel):
    Index: str
    UsingIP: str
    LongURL: str
    ShortURL: str
    Symbol: str
    Redirecting: str
    PrefixSuffix: str
    SubDomains: str
    HTTPS: str
    DomainRegLen: int
    Favicon: str
    NonStdPort: str
    HTTPSDomainURL: str
    RequestURL: str
    AnchorURL: str
    LinksInScriptTags: str
    ServerFormHandler: str
    InfoEmail: str
    AbnormalURL: str
    WebsiteForwarding: str
    StatusBarCust: str
    DisableRightClick: str
    UsingPopupWindow: str
    IframeRedirection: str
    AgeofDomain: str
    DNSRecording: str
    WebsiteTraffic: str
    PageRank: str
    GoogleIndex: str
    LinksPointingToPage: str
    StatsReport: str



@app.post("/predict")
def get_predict(user_data: ParmsData):
    series = pd.Series(user_data.model_dump())
    return {
            "prediction": manager.test_model.predict(series),
            "result map": "1 = Phishing Website, -1 = Non-Phishing Website"
            }


@app.get("/precision")
def get_precision():
    return {"model precision": manager.test_model.test_model()}



if __name__ == "__main__":
    uvicorn.run(app,    host="127.0.0.1", port=8000)
