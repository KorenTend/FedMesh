# strategies/scamoon.py (更新文件)

import torch
import copy
from .base_strategy import BaseStrategy
from .moon import MoonContrastiveLoss
from utils import get_model_state_dict, load_model_state_dict

class ScaMoon(BaseStrategy):
    """
    【已更新】采用分阶段训练逻辑，并与新的客户端实现兼容。
    聚合方式为FedAvg全量更新。
    """
    def __init__(self, model, config):
        super().__init__(model)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.contrastive_criterion = MoonContrastiveLoss(temperature=config['temperature']).to(self.device)
        self.previous_models = {}
        self.control_variate_global = {
            name: torch.zeros_like(param.detach().cpu()) 
            for name, param in get_model_state_dict(self.model).items()
        }
        self.warmup_rounds = self.config.get('warmup_rounds', 0)
        print(f"[ScaMoon] 策略已初始化。SCAFFOLD预热轮数: {self.warmup_rounds}")

    def client_train_step(self, client, round_num):
        cid = client.client_id
        global_weights = get_model_state_dict(self.model)

        global_model_copy = copy.deepcopy(self.model).to(self.device)
        load_model_state_dict(global_model_copy, global_weights)
        global_model_copy.eval()
        
        if cid not in self.previous_models:
            self.previous_models[cid] = copy.deepcopy(global_model_copy)
        previous_model_copy = self.previous_models[cid]
        previous_model_copy.eval()

        # --- 分阶段训练逻辑 ---
        original_mu = client.config['mu']
        # round_num 从0开始，所以用 round_num + 1 来比较
        if round_num + 1 <= self.warmup_rounds:
            if round_num + 1 == 1 or round_num + 1 == self.warmup_rounds:
                print(f"   (Round {round_num + 1}) >> 处于SCAFFOLD预热阶段, mu暂时设为0。")
            client.config['mu'] = 0.0
        else:
            if round_num + 1 == self.warmup_rounds + 1:
                print(f"   (Round {round_num + 1}) >> 预热结束, 启用MOON, mu恢复为{original_mu}。")
            client.config['mu'] = original_mu
        
        client.set_weights(global_weights)
        # 客户端返回 final_weights, control_delta
        final_weights, control_delta = client.train(
            algorithm='scamoon',
            contrastive_criterion=self.contrastive_criterion,
            global_model=global_model_copy,
            previous_model=previous_model_copy,
            control_variate_global=self.control_variate_global
        )
        
        # 恢复客户端原始config
        client.config['mu'] = original_mu

        # 直接使用返回的 final_weights 更新 previous_models
        with torch.no_grad():
            load_model_state_dict(self.previous_models[cid], final_weights)

        return final_weights, control_delta, len(client.dataset)

    def server_aggregate(self, client_updates, results=None):
        """服务器聚合逻辑: 全量更新 (FedAvg)"""
        print("   聚合客户端模型 (ScaMoon - Full Weight Aggregation)...")
        if not client_updates: return
        
        full_model_states = [update[0] for update in client_updates]
        delta_ccvs = [update[1] for update in client_updates]
        dataset_sizes = [update[2] for update in client_updates]
        total_samples = sum(dataset_sizes)
        if total_samples == 0: return

        # 1. 聚合模型权重 (标准的FedAvg方式)
        avg_model_state = {name: torch.zeros_like(val, device='cpu') for name, val in full_model_states[0].items()}
        for i, state_dict in enumerate(full_model_states):
            weight = dataset_sizes[i] / total_samples
            for name in avg_model_state:
                avg_model_state[name] += state_dict[name].cpu() * weight
        
        # 2. 聚合客户端控制变量增量
        avg_delta_ccv = {name: torch.zeros_like(val, device='cpu') for name, val in delta_ccvs[0].items()}
        for i, delta in enumerate(delta_ccvs):
            weight = dataset_sizes[i] / total_samples
            for name in avg_delta_ccv:
                avg_delta_ccv[name] += delta[name].cpu() * weight
        
        # 3. 更新全局模型和全局控制变量
        with torch.no_grad():
            load_model_state_dict(self.model, avg_model_state)
            for name in self.control_variate_global:
                 if name in avg_delta_ccv:
                    self.control_variate_global[name] += avg_delta_ccv[name]