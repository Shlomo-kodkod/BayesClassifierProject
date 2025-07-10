import pandas as pd

class NaiveBayes:
    
    def __init__(self, data:pd.DataFrame, target: str | int):
        self.__train_data = data
        self.__data_columns = [col for col in data.columns if col != target]
        self.__target_column = target
        self.__values_map = None
        self.__model_data = {}

    def  fit(self):
        self.__values_map = (self.__train_data[self.__target_column].value_counts() / self.__train_data[self.__target_column].count()).to_dict()
        
        for cls in self.__values_map:
            df_cls = self.__train_data[self.__train_data[self.__target_column] == cls] 
            self.__model_data[cls] = dict()

            for col in self.__data_columns:
                unique_val = self.__train_data[col].unique()
                val_count = df_cls[col].value_counts()

                prob_calc = dict()
                for val in unique_val:
                     prob_calc[val] = ((val_count.get(val, 0) + 1) / (len(df_cls) + len(unique_val)))
                
                self.__model_data[cls][col] = prob_calc
   
    @property
    def model_data(self) -> dict[any, dict[any, dict[any, float]]]:
        return self.__model_data
    
    @property
    def target_value(self) -> dict[str | int, float] | None:
        return self.__values_map
    
    @property
    def target_column(self) -> str | int:
        return self.__target_column

