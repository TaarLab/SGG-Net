import os
import csv

import zarr
import torch
import numpy as np
import networkx as nx
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv, AttentionalAggregation

from gs3d.utils.lru_cache import LRUCache
from gs3d.utils.model_utils import pc_to_graph_from_zarr

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)


class RewardNetwork(nn.Module):
    def __init__(self, node_feature_size=3, descriptor_size=64, action_size=5):
        super(RewardNetwork, self).__init__()

        # Graph attention layers
        self.gat1 = GATConv(node_feature_size, descriptor_size // 8, heads=8, concat=True)
        self.gat2 = GATConv(descriptor_size, descriptor_size // 8, heads=8, concat=True)
        self.gat3 = GATConv(descriptor_size, descriptor_size, heads=1, concat=True)

        # Attention pooling
        self.att_pool = AttentionalAggregation(nn.Linear(descriptor_size, 1))

        # Action processing with residual connections
        self.action_mlp = nn.Sequential(
            nn.Linear(action_size, 8 * action_size),
            nn.ReLU(),
            nn.BatchNorm1d(8 * action_size),
            nn.Linear(8 * action_size, 8 * action_size),
            nn.ReLU(),
            nn.BatchNorm1d(8 * action_size),
            nn.Linear(8 * action_size, descriptor_size),
            nn.ReLU(),
            nn.BatchNorm1d(descriptor_size),
        )

        # Fusion layers
        self.fc_combined = nn.Sequential(
            nn.Linear(2 * descriptor_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x, edge_index, batch_index, a):
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        x = F.relu(self.gat3(x, edge_index))

        # Attention pooling
        V = self.att_pool(x, batch_index)

        # Process action
        a = self.action_mlp(a)

        # Combine object and action representations
        x = torch.cat([V, a], dim=1)
        reward = self.fc_combined(x).squeeze(1)

        return reward



class GraspDataset(Dataset):
    def __init__(self, zarr_store):
        self.zarr_store = zarr_store
        self.trial_data_keys = [k for k in self.zarr_store.keys() if k.startswith('trial_data_')]

        # Flatten the dataset by creating an index mapping
        self.index_mapping = []
        for trial_idx, trial_key in enumerate(self.trial_data_keys):
            trial_data_array = self.zarr_store[trial_key][:]
            num_grasps = len(trial_data_array['grasp_pose'])
            self.index_mapping.extend([(trial_idx, grasp_idx) for grasp_idx in range(num_grasps)])

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        trial_idx, grasp_idx = self.index_mapping[idx]
        trial_data_key = self.trial_data_keys[trial_idx]

        # Load trial data from Zarr
        trial_data_array = self.zarr_store[trial_data_key][:]

        grasp_pose = torch.tensor(trial_data_array['grasp_pose'][grasp_idx], dtype=torch.float32).to(device)
        grasp_points = torch.tensor(trial_data_array['grasp_points'][grasp_idx], dtype=torch.float32).to(device)
        is_successful = trial_data_array['is_successful'][grasp_idx]
        expected_reward = torch.tensor(1.0 if is_successful else 0.0, dtype=torch.float32).to(device)

        pc_key = trial_data_key.replace('trial_data_', '')

        return pc_key, grasp_pose, grasp_points, expected_reward




def train_network(network, optimizer, zarr_store, cache, pc_keys, action, expected_reward):
    node_features, edge_index, batch_index = pc_to_graph_from_zarr(zarr_store, cache, pc_keys)
    action_tensor = action.to(device, non_blocking=True)
    expected_reward_tensor = expected_reward.to(device, dtype=torch.float32, non_blocking=True)
    node_features = torch.tensor(node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edge_index, device=device)
    batch_index = torch.tensor(batch_index, dtype=torch.int64, device=device)

    optimizer.zero_grad(set_to_none=True)

    # with torch.amp.autocast(device, cache_enabled=False):
    predicted_reward = network(node_features, edge_index, batch_index, action_tensor)
    loss = F.mse_loss(predicted_reward, expected_reward_tensor)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

    optimizer.step()

    return loss.item()


def main():
    cache = LRUCache(capacity=1000, zarr_path='cache.zarr')
    zarr_store = zarr.open('dataset.zarr', mode='r')
    dataset = GraspDataset(zarr_store)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    node_feature_size = 3  # Assuming point cloud has 3 features (x, y, z)
    action_size = 7  # Assuming action has 5 features
    reward_network = RewardNetwork(node_feature_size=node_feature_size, action_size=action_size).to(device)
    optimizer = torch.optim.AdamW(reward_network.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    reward_network.train()

    # Define directories for saving checkpoints, plots, and CSV
    checkpoint_dir = 'checkpoints'
    results_dir = 'results'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    best_loss = float('inf')
    num_epochs = 1000
    training_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        iteration = 0
        for pc_keys, grasp_pose, grasp_points, expected_reward in tqdm(data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            loss = train_network(reward_network, optimizer, zarr_store, cache, pc_keys, grasp_pose, expected_reward)
            epoch_loss += loss

            if iteration % 10 == 0:
                torch.cuda.empty_cache()

            iteration += 1
        avg_loss = epoch_loss / len(data_loader)
        training_history.append({'epoch': epoch + 1, 'loss': avg_loss})

        scheduler.step(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Training Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'reward_network_epoch_{epoch + 1}_loss_{avg_loss:.4f}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': reward_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    # Save the training history to CSV
    csv_path = os.path.join(results_dir, 'training_history.csv')
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['epoch', 'loss'])
        writer.writeheader()
        writer.writerows(training_history)
    print(f"Training history saved to {csv_path}")

    # Plot training loss history
    plt.figure()
    epochs = [entry['epoch'] for entry in training_history]
    losses = [entry['loss'] for entry in training_history]
    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plot_path = os.path.join(results_dir, 'training_loss_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training loss plot saved to {plot_path}")

if __name__ == "__main__":
    main()
