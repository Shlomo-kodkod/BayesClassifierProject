from src.utils.file_loader import FileLoader
from src.models.data_cleaner import DataCleaner
from src.models.naive_bayes import NaiveBayes
from src.models.test_model import TestModel



class Manager:
    def __init__(self):
        self.__data = None
        self.__model = None
        self.__test_model = None

    def load_data(self, file_path: str):
        data = FileLoader.load_data(file_path)
        clean_data = DataCleaner(data)
        clean_data.clean_data()
        self.__data = clean_data

    def create_model(self, target: str):
        trainer = NaiveBayes(self.__data.train_data, target)
        trainer.fit()
        self.__model = trainer

    def create_model_test(self):
        self.__test_model = TestModel(self.__data.test_data, self.__model)


    @property
    def model(self):
        return self.__model

    @property
    def test_model(self):
        return self.__test_model



