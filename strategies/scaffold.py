from .base_strategy import BaseStrategy
from utils import get_model_state_dict, load_model_state_dict
import torch

class SCAFFOLD(BaseStrategy):
    def __init__(self, model, config):
        super().__init__(model)
        self.config = config
        self.control_variate_global = {
            name: torch.zeros_like(param.detach().cpu()) 
            for name, param in get_model_state_dict(self.model).items()
        }

    def client_train_step(self, client, round_num):
        client.set_weights(get_model_state_dict(self.model))

        trained_weights, delta_ccv = client.train(
            algorithm='scaffold',
            control_variate_global=self.control_variate_global
        )
        return trained_weights, delta_ccv, len(client.dataset)

    def server_aggregate(self, client_updates, results=None):
        if not client_updates: return
        
        full_model_states = [update[0] for update in client_updates]
        delta_ccvs = [update[1] for update in client_updates]
        dataset_sizes = [update[2] for update in client_updates]
        total_samples = sum(dataset_sizes)
        
        if total_samples == 0: return
        avg_model_state = {name: torch.zeros_like(val, device='cpu') for name, val in full_model_states[0].items()}
        for i, state_dict in enumerate(full_model_states):
            weight = dataset_sizes[i] / total_samples
            for name in avg_model_state:
                avg_model_state[name] += state_dict[name].cpu() * weight

        avg_delta_ccv = {name: torch.zeros_like(val, device='cpu') for name, val in delta_ccvs[0].items()}
        for i, delta in enumerate(delta_ccvs):
            
            weight = dataset_sizes[i] / total_samples
            for name in avg_delta_ccv:
                avg_delta_ccv[name] += delta[name].cpu() * weight
        

        with torch.no_grad():

            load_model_state_dict(self.model, avg_model_state)

            for name in self.control_variate_global:
                 if name in avg_delta_ccv:
                    self.control_variate_global[name] += avg_delta_ccv[name]