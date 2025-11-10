from .base_strategy import BaseStrategy
from utils import get_model_state_dict, load_model_state_dict
import copy
import torch

class FedAvg(BaseStrategy):

    def client_train_step(self, client, round_num):
        client.set_weights(get_model_state_dict(self.model))
        return client.train()
        
    def server_aggregate(self, client_weights_list, dataset_sizes):
        total_samples = sum(dataset_sizes)
        global_weights = get_model_state_dict(self.model)
        avg_weights = copy.deepcopy(global_weights)

        for key in avg_weights:
            avg_weights[key] = avg_weights[key].cpu()

        for key in avg_weights.keys():
            if avg_weights[key].dtype.is_floating_point:
                avg_weights[key].zero_()
                for i, client_weights in enumerate(client_weights_list):
                    weight = dataset_sizes[i] / total_samples
                    avg_weights[key] += client_weights[key].cpu() * weight
            else: 
                avg_weights[key] = client_weights_list[0][key].cpu().clone()
        
        load_model_state_dict(self.model, avg_weights)