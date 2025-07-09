import pandas as pd
from data_loader import DataLoader 

class FileLoader(DataLoader):
        
    def load_data(self, file_path: str):
        try:
            if file_path.endswith(".csv"):
                return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error: {e}")
            return None

    
   


