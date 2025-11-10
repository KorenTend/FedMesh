# strategies/fedavg.py
from .base_strategy import BaseStrategy
from utils import get_model_state_dict, load_model_state_dict
import copy
import torch

class FedAvg(BaseStrategy):
    """标准的FedAvg算法策略"""

    def client_train_step(self, client, round_num):
        """FedAvg的客户端训练步骤很简单，就是常规训练"""
        client.set_weights(get_model_state_dict(self.model))
        return client.train()
        
    def server_aggregate(self, client_weights_list, dataset_sizes):
        """基于数据集大小进行加权平均"""
        print("   聚合客户端模型 (FedAvg)...")
        total_samples = sum(dataset_sizes)
        global_weights = get_model_state_dict(self.model)
        avg_weights = copy.deepcopy(global_weights)

        # 确保所有张量在CPU上进行聚合
        for key in avg_weights:
            avg_weights[key] = avg_weights[key].cpu()

        for key in avg_weights.keys():
            if avg_weights[key].dtype.is_floating_point:
                avg_weights[key].zero_()
                for i, client_weights in enumerate(client_weights_list):
                    weight = dataset_sizes[i] / total_samples
                    avg_weights[key] += client_weights[key].cpu() * weight
            else: 
                # 对于非浮点类型的参数（例如BN的num_batches_tracked），直接使用第一个客户端的
                avg_weights[key] = client_weights_list[0][key].cpu().clone()
        
        load_model_state_dict(self.model, avg_weights)