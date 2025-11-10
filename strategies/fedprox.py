# strategies/fedprox.py
from .fedavg import FedAvg
from utils import get_model_state_dict
import copy

class FedProx(FedAvg):
    """FedProx算法策略"""
    def __init__(self, model, config):
        super().__init__(model)
        self.config = config
        self.prox_mu = config.get('prox_mu', 0.01)

    def client_train_step(self, client, round_num):
        """FedProx的客户端训练，需要传递全局模型本身和mu参数"""
        print(f"   * FedProx (mu={self.prox_mu})")
        global_weights = get_model_state_dict(self.model)
        client.set_weights(global_weights)
        
        global_model_copy = copy.deepcopy(self.model)
        global_model_copy.eval()

        # 将全局模型和mu通过kwargs传递给客户端的train方法
        return client.train(
            prox_mu=self.prox_mu,
            global_model=global_model_copy
        )