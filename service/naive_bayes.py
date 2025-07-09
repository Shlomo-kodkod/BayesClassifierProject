import pandas as pd

class NaiveBayes:
    
    def __init__(self):
        self.__data_column = None
        self.__target_column = None
        self.__values_map = None
        self.model_data = {}

    def  fit(self, df:pd.DataFrame):
        self.__data_columns = df.columns[:-1]  
        self.__target_column = df.columns[-1]  
        self.__values_map = (df[self.__target_column].value_counts() / df[self.__target_column].count()).to_dict()
        

        for cls in self.__values_map:
            df_cls = df[df[self.__target_column] == cls] 
            self.__model_data[cls] = dict()

            for col in self.__data_columns:
                self.__model_data[cls][col] = ((df_cls[col].value_counts() + 1) / (df_cls[col].count() + len(df_cls[col].unique()))).to_dict()
        return self
    

