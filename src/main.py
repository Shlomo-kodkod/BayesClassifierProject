from utils.file_loader import FileLoader
from models.data_cleaner import DataCleaner
from models.naive_bayes import NaiveBayes
from models.test_model import TestModel



if __name__ == "__main__":
    df = FileLoader.load_data(r"data\buy_computer_data.csv")
    dc = DataCleaner(df)
    tr_df = dc.train_data
    ts_df = dc.test_data
    tr = NaiveBayes(tr_df)
    tr.fit()
    ts = TestModel(ts_df, tr.model_data, tr.target_value)
    res = ts.test_model()
    print(res)
