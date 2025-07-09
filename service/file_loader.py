import pandas as pd
from service.data_loader import DataLoader 

class FileLoader(DataLoader):
        
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame | None:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error: {e}")
            return None


