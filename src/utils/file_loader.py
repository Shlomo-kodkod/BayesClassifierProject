import pandas as pd
import os
from src.utils.data_loader import DataLoader

class FileLoader(DataLoader):
        
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame | None:
        try:
            if not os.path.isabs(file_path):
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                file_path = os.path.join(project_root, file_path)
            file_path = os.path.normpath(file_path)
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error: {e}")
            return None


