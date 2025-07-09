from abc import ABC, abstractmethod

class DataLoader(ABC):

    @abstractmethod
    def load_data(file_path: str):
        pass

