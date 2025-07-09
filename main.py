from service.file_loader import FileLoader
from service.data_cleaner import DataCleaner
from service.naive_bayes import NaiveBayes
from service.test_model import TestModel



if __name__ == "__main__":
    df = FileLoader.load_data(r"data\buy_computer_data.csv")
    print(df)
    dc = DataCleaner(df)
    dc.clean_data()
    print(dc)
    tr_df = dc.get_train_data()
    print(tr_df)
    ts_df = dc.get_test_data()
    print(ts_df)
    tr = NaiveBayes(tr_df)
    tr.fit()
    print(tr.model_data)
    print(tr.target_value)
    ts = TestModel(ts_df, tr.model_data, tr.target_value)
    res = ts.test_model()
    print(res)
