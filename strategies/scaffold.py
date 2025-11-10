# strategies/scaffold.py (完整替换)

from .base_strategy import BaseStrategy
from utils import get_model_state_dict, load_model_state_dict
import torch

class SCAFFOLD(BaseStrategy):
    """
    SCAFFOLD 算法策略 (根据新参考代码重构)
    核心变化:
    1. 客户端返回完整模型，服务器聚合方式变为FedAvg。
    2. 服务器控制变量(scv)的更新依赖于客户端控制变量增量(delta_ccv)的聚合。
    """
    def __init__(self, model, config):
        super().__init__(model)
        self.config = config
        
        # 初始化服务器控制变量 c (保持在CPU)
        self.control_variate_global = {
            name: torch.zeros_like(param.detach().cpu()) 
            for name, param in get_model_state_dict(self.model).items()
        }

    def client_train_step(self, client, round_num):
        """SCAFFOLD的客户端训练步骤"""
        client.set_weights(get_model_state_dict(self.model))

        # 客户端训练，并返回完整模型和控制变量增量
        trained_weights, delta_ccv = client.train(
            algorithm='scaffold',
            control_variate_global=self.control_variate_global
        )
        return trained_weights, delta_ccv, len(client.dataset)

    def server_aggregate(self, client_updates, results=None):
        """
        SCAFFOLD的服务器聚合
        NOTE: 此处的聚合逻辑参考了您提供的代码，它与SCAFFOLD原始论文有所不同。
              模型聚合方式为FedAvg，控制变量聚合方式为对客户端增量的加权平均。
        """
        print("   聚合客户端模型 (SCAFFOLD - Refactored)...")
        
        if not client_updates: return
        
        full_model_states = [update[0] for update in client_updates]
        delta_ccvs = [update[1] for update in client_updates]
        dataset_sizes = [update[2] for update in client_updates]
        total_samples = sum(dataset_sizes)
        
        if total_samples == 0: return

        # 1. 聚合模型权重 (FedAvg 方式)
        avg_model_state = {name: torch.zeros_like(val, device='cpu') for name, val in full_model_states[0].items()}
        for i, state_dict in enumerate(full_model_states):
            weight = dataset_sizes[i] / total_samples
            for name in avg_model_state:
                avg_model_state[name] += state_dict[name].cpu() * weight
        
        # 2. 聚合客户端控制变量增量
        # 参考代码实现: c_new = c_old + SUM(delta_c_i * w_i)
        # 注意：SCAFFOLD论文中是简单平均，但您的参考代码倾向于加权平均。我们在此遵循参考代码。
        avg_delta_ccv = {name: torch.zeros_like(val, device='cpu') for name, val in delta_ccvs[0].items()}
        for i, delta in enumerate(delta_ccvs):
            # 权重可以根据客户端数量简单平均，或根据数据量加权平均。此处使用简单平均更符合SCAFFOLD思想。
            # weight = 1.0 / len(delta_ccvs) 
            weight = dataset_sizes[i] / total_samples # 跟随参考代码的加权逻辑
            for name in avg_delta_ccv:
                avg_delta_ccv[name] += delta[name].cpu() * weight
        
        # 3. 更新全局模型和全局控制变量
        with torch.no_grad():
            # 加载新聚合的FedAvg模型
            load_model_state_dict(self.model, avg_model_state)

            # 更新服务器控制变量
            # c_new = c_old + avg_delta_c
            for name in self.control_variate_global:
                 if name in avg_delta_ccv:
                    self.control_variate_global[name] += avg_delta_ccv[name]