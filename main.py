import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse

from data_loader import prepare_datasets, pad_collate
from models import GridNet
from client import Client
from server import Server
from utils import TeeOutput, load_model_state_dict
from strategies.fedavg import FedAvg
from strategies.moon import MOON, HybridMoon
from strategies.fedprox import FedProx
from strategies.scaffold import SCAFFOLD
from strategies.scamoon import ScaMoon
import random
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(42)
    now = datetime.now()
    dir_timestamp = now.strftime('%y.%m.%d-%H-%M')
    run_name = f"{args.strategy}_{dir_timestamp}_E{args.local_epochs}_LR{args.learning_rate}_R{args.communication_rounds}"
    run_dir = Path("logs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_filename = run_dir / "training.log"
    log_file = open(log_filename, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(sys.stdout, log_file)

    try:
        print(f"     start FL training: {run_name}")
        print("=" * 60)
        print(f"parser: {vars(args)}")

        config = {
            'num_clients': 5,
            'communication_rounds': args.communication_rounds,
            'local_epochs': args.local_epochs,
            'batch_size': 32,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'features_to_use': [0, 1, 2, 3, 4, 5, 6, 7],
            'num_classes': 8,
            'mu': args.mu,
            'temperature': args.temperature,
            'drift_alpha': args.drift_alpha,
            'prox_mu': args.prox_mu,
            'warmup_rounds': args.warmup_rounds,
            'model_save_path': run_dir / "best_model.pt",
            'plot_save_path': run_dir / "metrics_plot.png"
        }
        
        client_group_map = {i: [2*i + 1, 2*i + 2] for i in range(config['num_clients'])}

        client_datasets, val_dataset, test_dataset = prepare_datasets(
            config['num_clients'], config['features_to_use'], client_group_map
        )
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=pad_collate, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=pad_collate, num_workers=4)
        print(f"\n   global value dataset: {len(val_dataset)}  | global test dataet: {len(test_dataset)} ")
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"\n device: {device}")
        
        global_model = GridNet(input_channels=len(config['features_to_use']), num_classes=config['num_classes'])
        print("\n init client...")
        clients = []
        for cid, local_dataset in client_datasets.items():
            print(f"   client {cid}: data groups {client_group_map[cid]}, number of dataset: {len(local_dataset)}")
            clients.append(Client(cid, global_model, local_dataset, config))

        if not clients:
            print("\n    none client available, exiting...")
            return
            
        print(f"\n strategy: {args.strategy}")
        strategy = None
        if args.strategy.lower() == 'fedavg': strategy = FedAvg(global_model)
        elif args.strategy.lower() == 'moon': strategy = MOON(global_model, config)
        elif args.strategy.lower() == 'fedprox': strategy = FedProx(global_model, config)
        elif args.strategy.lower() == 'scaffold': strategy = SCAFFOLD(global_model, config)
        elif args.strategy.lower() == 'scamoon': strategy = ScaMoon(global_model, config)
        else: raise ValueError(f"unknown strategy: {args.strategy}")

        server = Server(global_model, clients, strategy, val_loader, test_loader, config)
        val_losses, val_accuracies, best_val_accuracy = server.run()

        print("\n   testing the best model...")
        best_model_state_dict = torch.load(config['model_save_path'], map_location=device, weights_only=True)
        load_model_state_dict(global_model, best_model_state_dict)
        
        _, test_acc, test_report = server.evaluate_global_model(test_loader)
        test_macro_recall = test_report.get('macro avg', {}).get('recall', 0.0)
        
        print(f"   final test accuracy: {test_acc:.4f} | weight recall: {test_macro_recall:.4f} | best value accuracy: {best_val_accuracy:.4f}")
        
        print("\n   final evaluate (on global test dataset):")
        if test_report:
            for class_name, metrics in test_report.items():
                if isinstance(metrics, dict):
                    print(f"   - {class_name} Recall: {metrics.get('recall', 0.0):.2%}")

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1); plt.plot(val_losses, label='Global Validation Loss'); plt.title('Validation Loss vs. Rounds'); plt.xlabel('Round'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2); plt.plot(val_accuracies, label='Global Validation Accuracy'); plt.title('Validation Accuracy vs. Rounds'); plt.xlabel('Round'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(config['plot_save_path'], dpi=150)
        
    finally:
        print(f"\n log and results at: {run_dir}")
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    parser.add_argument('--strategy', type=str, default='ScaMoon', 
                        choices=['FedAvg', 'MOON', 'FedProx', 'SCAFFOLD','ScaMoon'], 
                        help='Federated learning algorithm strategy')
    parser.add_argument('--communication_rounds', type=int, default=200, help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=3, help='Number of local epochs for each client')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for client optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for client optimizer')
    
    parser.add_argument('--mu', type=float, default=0.5, help='Mu parameter for MOON')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for MOON contrastive loss')
    parser.add_argument('--warmup_rounds', type=int, default=50, help='Number of warmup rounds for ScaMoon/HybridMoon')
    
    parser.add_argument('--drift_alpha', type=float, default=0.1, help='Scaling factor alpha for ScaMoon')
    parser.add_argument('--prox_mu', type=float, default=0.01, help='Mu parameter for FedProx')
    
    args = parser.parse_args()
    main(args)