import pandas as pd

class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.__all_data = data
        self.__index = int(len(data) * 0.7)


    def clean_data(self):
        for col in self.__all_data.columns:
            self.__all_data[col] = self.__all_data[col].fillna(self.__all_data[col].value_counts().idxmax())

    def get_train_data(self):
        return self.__all_data[:self.__index]
            
    def get_test_data(self):
        return self.__all_data[self.__index:]
        
        