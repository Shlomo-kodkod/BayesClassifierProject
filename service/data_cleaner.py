import pandas as pd

class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.__index = int(len(self.__all_data) * 0.7)
        self.__all_data = data

    @staticmethod
    def clean_data():
        pass

    def get_train_data(self):
        if self.__index:
            return self.__all_data[:self.__index]
            

    def get_test_data(self):
        if self.__index:
            return self.__all_data[:self.__index]