from abc import ABC, abstractmethod

class DataLoader:

    @abstractmethod
    def data_loader(self):
        pass

    @abstractmethod
    def get_training_data(self):
        pass

    @abstractmethod
    def get_test_data(self):
        pass

