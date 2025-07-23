import pandas as pd
import logging
from src.data_loader.file_loader import FileLoader
from src.preprocessing.data_cleaner import DataCleaner
from src.models.naive_bayes import NaiveBayes
from src.models.test_model import TestModel
from src.models.predictor import Predictor

logger = logging.getLogger(__name__)

# Manager class for data loading, cleaning, model training, and testing
class Manager:
    def __init__(self):
        self.__data = None
        self.__clean_data = None
        self.__model = None
        self.__test_model = None

    # Load data from a file using FileLoader
    def load_data(self, path: str):
        logger.info(f"Loading data from {path}")
        data = FileLoader.load_data(path)
        logger.info("Data loaded successfully")
        self.__data = data

    # Clean the data using DataCleaner
    def data_cleaner(self, df):
        logger.info("Starting data cleaning")
        clean_data = DataCleaner(df)
        clean_data.clean_data()
        self.__clean_data = clean_data
        logger.info("Data cleaning completed")

    # Train a NaiveBayes model with cleaned training data
    def create_model(self, path: str, target: str):
        self.load_data(path)
        self.data_cleaner(self.data)
        logger.info(f"Training NaiveBayes model with target: {target}")
        trainer = NaiveBayes(self.__clean_data.train_data, target)
        trainer.fit()
        self.__model = trainer
        logger.info("Model training completed")

    # Create a TestModel instance for evaluating the model
    def create_model_test(self):
        logger.info("Creating TestModel instance")
        self.__test_model = TestModel(self.__clean_data.test_data, self.__model)
        logger.info("TestModel instance created")

    # Return the raw loaded data
    @property
    def data(self):
        return self.__data

    # Return the trained model
    @property
    def model_data(self):
        return self.__model


    # Return the test model instance
    @property
    def test_model(self):
        return self.__test_model

    