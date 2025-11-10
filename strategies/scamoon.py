import torch
import copy
from .base_strategy import BaseStrategy
from .moon import MoonContrastiveLoss
from utils import get_model_state_dict, load_model_state_dict

class ScaMoon(BaseStrategy):

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

        original_mu = client.config['mu']
        if round_num + 1 <= self.warmup_rounds:
            if round_num + 1 == 1 or round_num + 1 == self.warmup_rounds:
                print(f"   (Round {round_num + 1}) >> in warmup stage , mu: 0。")
            client.config['mu'] = 0.0
        else:
            if round_num + 1 == self.warmup_rounds + 1:
                print(f"   (Round {round_num + 1}) >> warmup end, mu:{original_mu}。")
            client.config['mu'] = original_mu
        
        client.set_weights(global_weights)
        final_weights, control_delta = client.train(
            algorithm='scamoon',
            contrastive_criterion=self.contrastive_criterion,
            global_model=global_model_copy,
            previous_model=previous_model_copy,
            control_variate_global=self.control_variate_global
        )
        
       
        client.config['mu'] = original_mu

       
        with torch.no_grad():
            load_model_state_dict(self.previous_models[cid], final_weights)

        return final_weights, control_delta, len(client.dataset)

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