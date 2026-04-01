from .base_strategy import BaseStrategy
from utils import get_model_state_dict, load_model_state_dict
import torch

class SCAFFOLD(BaseStrategy):
    def __init__(self, model, config):

    def client_train_step(self, client, round_num):
        client.set_weights(get_model_state_dict(self.model))

        trained_weights, delta_ccv = client.train(
            algorithm='scaffold',
            control_variate_global=self.control_variate_global
        )
        return trained_weights, delta_ccv, len(client.dataset)


        

        with torch.no_grad():

            load_model_state_dict(self.model, avg_model_state)

            for name in self.control_variate_global:
                 if name in avg_delta_ccv:
                    self.control_variate_global[name] += avg_delta_ccv[name]
