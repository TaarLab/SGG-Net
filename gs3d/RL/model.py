import os
import re
import csv

import zarr
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, global_max_pool
from torch.utils.data import Dataset, DataLoader, random_split

from gs3d.utils.lru_cache import LRUCache
from gs3d.utils.model_utils import pc_to_graph_from_zarr

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)


class RewardNetwork(nn.Module):
    def __init__(self, node_feature_size=3, descriptor_size=64, action_size=7, fc_size=64):
        super(RewardNetwork, self).__init__()

        # Graph convolution layers
        self.initial_conv = GCNConv(node_feature_size, descriptor_size)
        self.conv1 = GCNConv(descriptor_size, descriptor_size)
        self.conv2 = GCNConv(descriptor_size, descriptor_size)
        self.conv3 = GCNConv(descriptor_size, descriptor_size)

        # Layer normalization
        self.ln1 = nn.LayerNorm(descriptor_size)
        self.ln2 = nn.LayerNorm(descriptor_size)
        self.ln3 = nn.LayerNorm(4 * action_size)
        self.ln4 = nn.LayerNorm(fc_size)
        self.ln5 = nn.LayerNorm(8 * action_size)

        # Action fully connected layers
        self.afc1 = nn.Linear(action_size, 8 * action_size)
        self.afc2 = nn.Linear(8 * action_size, 8 * action_size)
        self.afc3 = nn.Linear(8 * action_size, descriptor_size)

        # Final fully connected layers
        self.fc1 = nn.Linear(descriptor_size, fc_size)
        self.fc2 = nn.Linear(fc_size, 1)

        # Xavier initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.afc1.weight)
        nn.init.xavier_uniform_(self.afc2.weight)
        nn.init.xavier_uniform_(self.afc3.weight)

    def forward(self, x, edge_index, batch_index, action):
        hidden0 = self.initial_conv(x, edge_index)

        hidden1 = self._gcn_block(hidden0, self.conv1, edge_index)
        hidden2 = self._gcn_block(hidden1, self.conv2, edge_index)
        hidden3 = self._gcn_block(hidden2, self.conv3, edge_index)

        x = global_max_pool(hidden3, batch_index)
        V = self.ln2(x)

        action = self._action_block(action)

        x = V * action
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.ln4(x)
        x = self.fc2(x)

        return x.squeeze(1)

    def _gcn_block(self, x, conv_layer, edge_index):
        out = conv_layer(x, edge_index)
        out = F.leaky_relu(out)
        return out + x

    def _action_block(self, action):
        action = self.afc1(action)
        action = F.leaky_relu(action)
        action = self.ln5(action)
        action = self.afc2(action)
        action = F.leaky_relu(action)
        action = self.afc3(action)
        action = F.leaky_relu(action)
        return self.ln1(action)


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
        expected_reward = torch.tensor(1.0 if is_successful else -1.0, dtype=torch.float32).to(device)

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

    predicted_reward = network(node_features, edge_index, batch_index, action_tensor)
    loss = F.mse_loss(predicted_reward, expected_reward_tensor)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

    optimizer.step()

    return loss.item()

def evaluate_network(network, zarr_store, cache, pc_keys, action, expected_reward):
    node_features, edge_index, batch_index = pc_to_graph_from_zarr(zarr_store, cache, pc_keys)
    action_tensor = action.to(device, non_blocking=True)
    expected_reward_tensor = expected_reward.to(device, dtype=torch.float32, non_blocking=True)
    node_features = torch.tensor(node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edge_index, device=device)
    batch_index = torch.tensor(batch_index, dtype=torch.int64, device=device)

    predicted_reward = network(node_features, edge_index, batch_index, action_tensor)
    loss = F.mse_loss(predicted_reward, expected_reward_tensor)

    return loss.item()


def main():
    cache = LRUCache(capacity=1000, zarr_path='cache.zarr')
    zarr_store = zarr.open('dataset.zarr', mode='r')
    dataset = GraspDataset(zarr_store)

    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    node_feature_size = 3  # Assuming point cloud has 3 features (x, y, z)
    action_size = 7  # Assuming action has 5 features
    reward_network = RewardNetwork(node_feature_size=node_feature_size, action_size=action_size).to(device)
    optimizer = torch.optim.AdamW(reward_network.parameters(), lr=1e-2, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=25)

    # Define directories for saving checkpoints, plots, and CSV
    checkpoint_dir = 'checkpoints'
    results_dir = 'results'
    start_epoch = 0
    num_epochs = 1000
    training_history = []
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, 'training_history.csv')

    if os.path.exists(csv_path):
        with open(csv_path, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            training_history = [row for row in reader]
            if training_history:
                last_logged_epoch = int(training_history[-1]['epoch'])
                start_epoch = last_logged_epoch

    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    if checkpoint_files:
        latest_checkpoint_path = None
        latest_epoch = -1
        epoch_pattern = re.compile(r'reward_network_epoch_(\d+)_loss_.*\.pth')

        for checkpoint_file in checkpoint_files:
            match = epoch_pattern.match(checkpoint_file)
            if match:
                checkpoint_epoch = int(match.group(1))
                if checkpoint_epoch > latest_epoch:
                    latest_epoch = checkpoint_epoch
                    latest_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

        if latest_checkpoint_path is not None:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            reward_network.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']
            print(f"Resumed training from checkpoint {latest_checkpoint_path}, starting at epoch {start_epoch}, best loss {best_loss:.4f}")


    with open(csv_path, mode='a', newline='') as csv_file:
        fieldnames = ['epoch', 'loss', 'val_loss', 'lr']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not training_history:
            writer.writeheader()

        for epoch in range(start_epoch, num_epochs):
            train_loss = 0
            iteration = 0
            reward_network.train()
            for pc_keys, grasp_pose, grasp_points, expected_reward in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                loss = train_network(reward_network, optimizer, zarr_store, cache, pc_keys, grasp_pose, expected_reward)
                train_loss += loss

                if iteration % 5 == 0:
                    torch.cuda.empty_cache()

                iteration += 1

            avg_train_loss = train_loss / len(train_loader)

            # scheduler.step(avg_train_loss)
            scheduler.step()

            val_loss = 0
            reward_network.eval()
            with torch.no_grad():
                for pc_keys, grasp_pose, grasp_points, expected_reward in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Validation]'):
                    loss = evaluate_network(reward_network, zarr_store, cache, pc_keys, grasp_pose, expected_reward)
                    val_loss += loss
            avg_val_loss = val_loss / len(val_loader)

            training_history.append({'epoch': epoch + 1, 'loss': avg_train_loss, 'val_loss': avg_val_loss})
            writer.writerow({'epoch': epoch + 1, 'loss': avg_train_loss, 'val_loss': avg_val_loss, 'lr': scheduler.get_last_lr()[0]})
            csv_file.flush()

            print(f"Epoch [{epoch + 1}/{num_epochs}] Average Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

            checkpoint_path = os.path.join(checkpoint_dir, f'reward_network_epoch_{epoch + 1}_loss_{avg_train_loss:.4f}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': reward_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

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
