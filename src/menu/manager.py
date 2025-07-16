from src.utils.file_loader import FileLoader
from src.models.data_cleaner import DataCleaner
from src.models.naive_bayes import NaiveBayes
from src.models.test_model import TestModel

# Manager class for data loading, cleaning, model training, and testing
class Manager:
    def __init__(self):
        self.__data = None
        self.__clean_data = None
        self.__model = None
        self.__test_model = None

    # Load data from a file using FileLoader
    def load_data(self, file_path: str):
        data = FileLoader.load_data(file_path)
        self.__data = data

    # Clean the data using DataCleaner
    def clean_data(self, df):
        clean_data = DataCleaner(df)
        clean_data.clean_data()
        self.__clean_data = clean_data

    # Train a NaiveBayes model with cleaned training data
    def create_model(self, target: str):
        trainer = NaiveBayes(self.__clean_data.train_data, target)
        trainer.fit()
        self.__model = trainer

    # Create a TestModel instance for evaluating the model
    def create_model_test(self):
        self.__test_model = TestModel(self.__clean_data.test_data, self.__model)

    # Return the raw loaded data
    @property
    def data(self):
        return self.__data

    # Return the trained model
    @property
    def model(self):
        return self.__model

    # Return the test model instance
    @property
    def test_model(self):
        return self.__test_model



