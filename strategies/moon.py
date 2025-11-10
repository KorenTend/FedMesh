# strategies/moon.py
import torch
import torch.nn as nn
from .fedavg import FedAvg # MOON的聚合方式与FedAvg相同，因此可以继承
from utils import get_model_state_dict, load_model_state_dict
import copy

class MoonContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(MoonContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, rep_curr, rep_pos, rep_neg):
        rep_curr_flat = rep_curr.view(rep_curr.shape[0], -1)
        rep_pos_flat = rep_pos.view(rep_pos.shape[0], -1)
        rep_neg_flat = rep_neg.view(rep_neg.shape[0], -1)

        pos_sim = self.cosine_similarity(rep_curr_flat, rep_pos_flat) / self.temperature
        neg_sim = self.cosine_similarity(rep_curr_flat, rep_neg_flat) / self.temperature
        
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        loss = -pos_sim + torch.logsumexp(logits, dim=1)
        
        return loss.mean()

class MOON(FedAvg):
    """MOON算法策略"""
    def __init__(self, model, config):
        super().__init__(model)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.contrastive_criterion = MoonContrastiveLoss(temperature=config['temperature']).to(self.device)
        self.previous_models = {} # 存储每个客户端上一轮的模型

    def client_train_step(self, client, round_num):
        """MOON的客户端训练步骤，需要传递全局模型和上一轮的本地模型"""
        cid = client.client_id
        global_weights = get_model_state_dict(self.model)

        # 准备 MOON 需要的模型
        global_model_copy = copy.deepcopy(self.model).to(self.device)
        global_model_copy.eval()
        load_model_state_dict(global_model_copy, global_weights)
        
        if cid not in self.previous_models:
            # 如果是第一轮，上一轮模型就是当前全局模型
            self.previous_models[cid] = copy.deepcopy(global_model_copy)

        previous_model_copy = self.previous_models[cid]
        previous_model_copy.eval()

        # 更新客户端权重，并开始训练
        client.set_weights(global_weights)
        
        # 将额外的模型和损失函数通过kwargs传递给客户端的train方法
        trained_weights = client.train(
            contrastive_criterion=self.contrastive_criterion,
            global_model=global_model_copy,
            previous_model=previous_model_copy
        )

        # 训练结束后，更新该客户端的"上一轮模型"
        load_model_state_dict(self.previous_models[cid], trained_weights)
        
        return trained_weights

class HybridMoon(MOON):
    """先用FedAvg预热，再切换到MOON的混合策略"""
    def __init__(self, model, config):
        super().__init__(model, config)
        self.warmup_rounds = config.get('warmup_rounds', 0)
        self.warmup_strategy = FedAvg(model) # 创建一个FedAvg实例用于预热阶段

    def client_train_step(self, client, round_num):
        if round_num < self.warmup_rounds:
            if round_num == 0:
                print(f"   🔥 Phase 1 (Round {round_num+1}): Training with FedAvg...")
            # 在预热阶段，调用纯FedAvg的训练步骤
            return self.warmup_strategy.client_train_step(client, round_num)
        else:
            if round_num == self.warmup_rounds:
                 print(f"   🌙 Phase 2 (Round {round_num+1}): Switching to MOON algorithm...")
            # 预热结束后，调用MOON的训练步骤
            return super().client_train_step(client, round_num)