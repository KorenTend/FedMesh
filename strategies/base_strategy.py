from abc import ABC, abstractmethod

class BaseStrategy(ABC):    
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def server_aggregate(self, client_updates, results):
        pass

    @abstractmethod
    def client_train_step(self, client, round_num):
        pass