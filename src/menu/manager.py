from src.utils.file_loader import FileLoader
from src.models.data_cleaner import DataCleaner
from src.models.naive_bayes import NaiveBayes
from src.models.test_model import TestModel



class Manager:
    def __init__(self, file_path, target):
        self.__data = FileLoader.load_data(file_path)
        self.__clean_data = DataCleaner(self.__data)
        self.__clean_data.clean_data()
        self.__trainer = NaiveBayes(self.__clean_data.train_data, target)
        self.__trainer.fit()
        self.__test_model = TestModel(self.__clean_data.test_data, self.__trainer)

    @property
    def model(self):
        return self.__trainer

    @property
    def test_model(self):
        return self.__test_model



