import torch
import torch.nn as nn
from .fedavg import FedAvg
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
    def __init__(self, model, config):
        super().__init__(model)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.contrastive_criterion = MoonContrastiveLoss(temperature=config['temperature']).to(self.device)
        self.previous_models = {} 

    def client_train_step(self, client, round_num):

        cid = client.client_id
        global_weights = get_model_state_dict(self.model)


        global_model_copy = copy.deepcopy(self.model).to(self.device)
        global_model_copy.eval()
        load_model_state_dict(global_model_copy, global_weights)
        
        if cid not in self.previous_models:

            self.previous_models[cid] = copy.deepcopy(global_model_copy)

        previous_model_copy = self.previous_models[cid]
        previous_model_copy.eval()

        client.set_weights(global_weights)
        
        trained_weights = client.train(
            contrastive_criterion=self.contrastive_criterion,
            global_model=global_model_copy,
            previous_model=previous_model_copy
        )
        load_model_state_dict(self.previous_models[cid], trained_weights)
        
        return trained_weights