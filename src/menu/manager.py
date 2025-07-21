from data_loader.file_loader import FileLoader
from models.data_cleaner import DataCleaner
from models.naive_bayes import NaiveBayes
from models.test_model import TestModel
from models.predictor import Predictor
import logging

logger = logging.getLogger(__name__)

# Manager class for data loading, cleaning, model training, and testing
class Manager:
    def __init__(self):
        self.__data = None
        self.__clean_data = None
        self.__model = None
        self.__test_model = None
        self.__predictor = None

    # Load data from a file using FileLoader
    def load_data(self, file_path: str):
        logger.info(f"Loading data from {file_path}")
        data = FileLoader.load_data(file_path)
        self.__data = data
        logger.info("Data loaded successfully")

    # Clean the data using DataCleaner
    def clean_data(self, df):
        logger.info("Starting data cleaning")
        clean_data = DataCleaner(df)
        clean_data.clean_data()
        self.__clean_data = clean_data
        logger.info("Data cleaning completed")

    # Train a NaiveBayes model with cleaned training data
    def create_model(self, target: str):
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

    def create_predictor(self):
        logger.info("Creating Predictor instance")
        self.__predictor = Predictor(self.__model)
        logger.info("Predictor instance created")

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

    # Return the predictor instance
    @property
    def predictor(self):
        return self.__predictor



