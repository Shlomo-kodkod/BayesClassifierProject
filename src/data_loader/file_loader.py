import pandas as pd
import os
import logging
from data_loader.data_loader import DataLoader

logger = logging.getLogger(__name__)

# FileLoader load CSV files from disk
class FileLoader(DataLoader):

    # Build an absolute path from a relative path 
    @staticmethod
    def build_abs_path(file_path):
        if not os.path.isabs(rf"{file_path}"):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            file_path = os.path.join(project_root, file_path)
        return os.path.normpath(file_path)

    # Load a CSV file into a DataFrame
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame | None:
        try:
            file_path = FileLoader.build_abs_path(file_path)
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            return None


