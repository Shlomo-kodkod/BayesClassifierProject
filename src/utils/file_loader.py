import pandas as pd
import os
from src.utils.data_loader import DataLoader

class FileLoader(DataLoader):
        
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame | None:
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            abs_path = os.path.join(project_root, "data", "buy_computer_data.csv")
            return pd.read_csv(abs_path)
        except Exception as e:
            print(f"Error: {e}")
            return None


