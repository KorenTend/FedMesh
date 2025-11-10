import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path

class MultiSizeMeshDataset(Dataset):
    def __init__(self, base_dir="./cnn_input_data/", group_indices=None, feature_indices=None):
        self.base_dir = Path(base_dir)
        self.samples = []
        self.feature_indices = feature_indices

        if group_indices is None:
            group_indices = range(1, 11)

        for idx in group_indices:
            group_dirs = list(self.base_dir.glob(f"group{idx}_8x*"))
            if not group_dirs:
                print(f"    not found {idx}，skip")
                continue

            group_dir = group_dirs[0]
            batch_file = group_dir / "batch_data.pt"

            if batch_file.exists():
                print(f"   loading {idx}: {group_dir.name}")
                batch_data = torch.load(batch_file, map_location="cpu")
                inputs, labels = batch_data["inputs"], batch_data["labels"]
                for x, y in zip(inputs, labels):
                    if self.feature_indices:
                        x = x[self.feature_indices]
                    self.samples.append((x, y))
            else:
                print(f"  group {idx}'s batch_data.pt doesnt exit, skip")
        print(f"   load complete, total: {len(self.samples)}。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def pad_collate(batch):
    inputs, labels = zip(*batch)
    c, h_max, w_max = inputs[0].shape[0], max(x.shape[1] for x in inputs), max(x.shape[2] for x in inputs)
    padded_inputs = [F.pad(x, (0, w_max - x.shape[2], 0, h_max - x.shape[1])) for x in inputs]
    return torch.stack(padded_inputs), torch.tensor(labels)

def prepare_datasets(num_clients, feature_indices, client_group_map):
    print("\n dataset prepare : train (80%)/ value (10%)/ test (10%)...")
    client_train_datasets = {i: [] for i in range(num_clients)}
    server_val_datasets, server_test_datasets = [], []

    for group_id in range(1, 11):
        group_dataset = MultiSizeMeshDataset(base_dir="./cnn_input_data/", group_indices=[group_id], feature_indices=feature_indices)
        if len(group_dataset) == 0: continue
        
        train_indices, temp_indices = train_test_split(range(len(group_dataset)), test_size=0.2, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
        
        server_val_datasets.append(Subset(group_dataset, val_indices))
        server_test_datasets.append(Subset(group_dataset, test_indices))
        
        for client_id, groups in client_group_map.items():
            if group_id in groups:
                client_train_datasets[client_id].append(Subset(group_dataset, train_indices))
                break
    
    final_client_datasets = {}
    for cid, datasets in client_train_datasets.items():
        if datasets:
            final_client_datasets[cid] = ConcatDataset(datasets)

    global_val_dataset = ConcatDataset(server_val_datasets)
    global_test_dataset = ConcatDataset(server_test_datasets)
    
    return final_client_datasets, global_val_dataset, global_test_dataset