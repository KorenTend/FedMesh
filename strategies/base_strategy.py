# strategies/base_strategy.py
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """所有联邦学习算法策略的抽象基类"""
    
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def server_aggregate(self, client_updates, results):
        """
        在服务器端聚合客户端模型。
        - client_updates: 客户端上传的权重列表
        - results: 客户端的数据集大小等信息
        """
        pass

    @abstractmethod
    def client_train_step(self, client, round_num):
        """
        定义客户端的训练步骤。
        这允许策略向客户端传递额外的信息（例如MOON中的旧模型）。
        """
        pass