from src.utils.file_loader import FileLoader
from src.models.data_cleaner import DataCleaner
from src.models.naive_bayes import NaiveBayes
from src.models.test_model import TestModel



class Manager:
    def __init__(self):
        self.__data = None
        self.__clean_data = None
        self.__model = None
        self.__test_model = None

    def load_data(self, file_path: str):
        data = FileLoader.load_data(file_path)
        self.__data = data

    def clean_data(self, df):
        clean_data = DataCleaner(df)
        clean_data.clean_data()
        self.__clean_data = clean_data


    def create_model(self, target: str):
        trainer = NaiveBayes(self.__clean_data.train_data, target)
        trainer.fit()
        self.__model = trainer

    def create_model_test(self):
        self.__test_model = TestModel(self.__clean_data.test_data, self.__model)


    @property
    def data(self):
        return self.__data

    @property
    def model(self):
        return self.__model

    @property
    def test_model(self):
        return self.__test_model



