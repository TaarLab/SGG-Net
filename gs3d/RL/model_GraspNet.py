import json
import os
import re
import csv

import torch
import torch.nn as nn
import zarr
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch.utils.data import Dataset, DataLoader, random_split

from graspnetAPI import GraspNet
from gs3d.utils.lru_cache import LRUCache
from gs3d.utils.model_utils import pc_to_graph_from_graspnet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)


class RewardNetwork(nn.Module):
    def __init__(self, node_feature_size=3, descriptor_size=256, action_size=7, fc_size=64):
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

    def forward(self, x, edge_index, batch_index, a):
        hidden0 = self.initial_conv(x, edge_index)

        hidden1 = self._gcn_block(hidden0, self.conv1, edge_index)
        hidden2 = self._gcn_block(hidden1, self.conv2, edge_index)
        hidden3 = self._gcn_block(hidden2, self.conv3, edge_index)

        x = global_max_pool(hidden3, batch_index)
        V = self.ln2(x)

        a = self._action_block(a)

        x = V * a
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.ln4(x)
        x = self.fc2(x)

        return x.squeeze(1)

    def _gcn_block(self, x, conv_layer, edge_index):
        out = conv_layer(x, edge_index)
        out = F.leaky_relu(out)
        return out + x

    def _action_block(self, a):
        a = self.afc1(a)
        a = F.leaky_relu(a)
        a = self.ln5(a)
        a = self.afc2(a)
        a = F.leaky_relu(a)
        a = self.afc3(a)
        a = F.leaky_relu(a)
        return self.ln1(a)


class GraspDataset(Dataset):
    def __init__(self, zarr_store):
        self.zarr_store = zarr_store
        self.device = device

        self.index_map = []
        for group_name in self.zarr_store.array_keys():
            scene_number, ann_id = map(int, group_name.split('_'))
            self.index_map.append((scene_number, ann_id))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        scene_number, ann_id = self.index_map[idx]

        data_chunk = self.zarr_store[f"{scene_number}_{ann_id}"][:]

        data_chunk_tensor = torch.tensor(data_chunk, dtype=torch.float32).to(self.device)

        grasp_poses_tensor = torch.cat((data_chunk_tensor[:, 3:6], data_chunk_tensor[:, 6:10]), dim=1)
        expected_rewards_tensor = data_chunk_tensor[:, 10]

        pc_key = f"{scene_number}-{ann_id}"

        return pc_key, grasp_poses_tensor, expected_rewards_tensor.to(self.device)


def train_network(network, optimizer, cache, grasp_net, pc_keys, action, expected_reward):
    total_loss = 0.0
    total_samples = 0

    for i in range(len(pc_keys)):
        pc_key = pc_keys[i]
        grasp_poses = action[i]
        expected_rewards = expected_reward[i]

        # Convert grasp poses and rewards to tensors on the device
        grasp_poses_tensor = torch.tensor(grasp_poses, dtype=torch.float32, device=device)
        expected_rewards_tensor = torch.tensor(expected_rewards, dtype=torch.float32, device=device)

        # Get node features, edge index, and batch index for GNN
        node_features, edge_index, batch_index = pc_to_graph_from_graspnet(cache, grasp_net, [pc_key])

        # Move tensors to the device
        node_features = torch.tensor(node_features, dtype=torch.float32, device=device)
        edge_index = torch.tensor(edge_index, device=device)
        batch_index = torch.tensor(batch_index, dtype=torch.int64, device=device)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass through the network
        predicted_reward = network(node_features, edge_index, batch_index, grasp_poses_tensor)
        loss = F.mse_loss(predicted_reward, expected_rewards_tensor)

        # Backpropagation and optimization step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate loss and count
        total_loss += loss.item()
        total_samples += 1

    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return mean_loss

def evaluate_network(network, cache, grasp_net, pc_keys, action, expected_reward):
    total_loss = 0.0
    total_samples = 0
    for i in range(len(pc_keys)):
        pc_key = pc_keys[i]
        grasp_poses = action[i]
        expected_rewards = expected_reward[i]

        # Convert grasp poses and rewards to tensors on the device
        grasp_poses_tensor = torch.tensor(grasp_poses, dtype=torch.float32, device=device)
        expected_rewards_tensor = torch.tensor(expected_rewards, dtype=torch.float32, device=device)

        # Get node features, edge index, and batch index for GNN
        node_features, edge_index, batch_index = pc_to_graph_from_graspnet(cache, grasp_net, [pc_key])

        # Move tensors to the device
        node_features = torch.tensor(node_features, dtype=torch.float32, device=device)
        edge_index = torch.tensor(edge_index, device=device)
        batch_index = torch.tensor(batch_index, dtype=torch.int64, device=device)

        # Forward pass through the network
        predicted_reward = network(node_features, edge_index, batch_index, grasp_poses_tensor)
        loss = F.mse_loss(predicted_reward, expected_rewards_tensor)

        # Accumulate loss and count
        total_loss += loss.item()
        total_samples += 1

    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return mean_loss


def main():
    cache = LRUCache(capacity=1000, zarr_path='cache.zarr')
    grasp_net_dataset = zarr.open('grasp_net_dataset.zarr', mode='r')
    with open('pathes.json', 'r') as f:
        config = json.load(f)
    graspnet_root = config['graspnet_root']
    grasp_net = GraspNet(graspnet_root, camera="kinect", split='custom')
    dataset = GraspDataset(grasp_net_dataset)

    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    def custom_collate_fn(batch):
        pc_keys, grasp_poses, expected_rewards = zip(*batch)
        return pc_keys, grasp_poses, expected_rewards
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, collate_fn=custom_collate_fn)

    node_feature_size = 3
    action_size = 7
    reward_network = RewardNetwork(node_feature_size=node_feature_size, action_size=action_size).to(device)
    optimizer = torch.optim.AdamW(reward_network.parameters(), lr=1e-2, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
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
            for pc_keys, grasp_pose, expected_reward in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                loss = train_network(reward_network, optimizer, cache, grasp_net, pc_keys, grasp_pose, expected_reward)
                train_loss += loss

                if iteration % 25 == 0:
                    torch.cuda.empty_cache()

                iteration += 1

            avg_train_loss = train_loss / len(train_loader)

            # scheduler.step(avg_train_loss)
            scheduler.step()

            val_loss = 0
            reward_network.eval()
            with torch.no_grad():
                for pc_keys, grasp_pose, expected_reward in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Validation]'):
                    loss = evaluate_network(reward_network, cache, grasp_net, pc_keys, grasp_pose, expected_reward)
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

if __name__ == "__main__":
    main()
