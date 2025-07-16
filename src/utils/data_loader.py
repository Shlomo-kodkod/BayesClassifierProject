from abc import ABC, abstractmethod

# Abstract base class for data loaders
class DataLoader(ABC):
    @staticmethod
    @abstractmethod
    def load_data(file_path: str):
        pass

