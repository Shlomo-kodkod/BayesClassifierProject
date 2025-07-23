import pandas as pd


# Class for cleaning the data and splitting it into train and test sets
class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.__all_data = data
        self.__index = int(len(data) * 0.7)

    # Fill missing values in each column with the most frequent value
    def clean_data(self):
        for col in self.__all_data.columns:
            num_missing = self.__all_data[col].isna().sum()
            if num_missing > 0:
                self.__all_data[col] = self.__all_data[col].fillna(self.__all_data[col].value_counts().idxmax())

    # Return the training set 
    @property
    def train_data(self) -> pd.DataFrame:
        return self.__all_data[:self.__index]

     # Return the test set     
    @property
    def test_data(self) -> pd.DataFrame:
        return self.__all_data[self.__index:]
        
        