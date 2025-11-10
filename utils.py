# utils.py
import sys
import torch
import torch.nn as nn

class TeeOutput:
    """A helper class to redirect print output to both console and a file."""
    def __init__(self, *files):
        self.files = files
    
    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()
    
    def flush(self):
        for file in self.files:
            file.flush()

def get_model_state_dict(model):
    """安全地从模型中获取state_dict，处理DataParallel包装器。"""
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

def load_model_state_dict(model, state_dict):
    """安全地将state_dict加载到模型中，处理DataParallel包装器。"""
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict,strict=False)