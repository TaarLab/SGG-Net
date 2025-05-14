import json
import torch
import numpy as np
import zarr
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from graspnetAPI import GraspNet
from gs3d.utils.lru_cache import LRUCache
from gs3d.utils.model_utils import pc_to_graph_from_graspnet
from gs3d.RL.model_GraspNet import GraspDataset
# from gs3d.RL.model import RewardNetwork
from gs3d.RL.model_GraspNet import RewardNetwork
# from gs3d.gtg import RewardNetwork

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_network(network, cache, grasp_net, pc_keys, action, expected_reward):
    """
    Modified evaluation function to return predictions and actuals
    """
    node_features, edge_index, batch_index = pc_to_graph_from_graspnet(cache, grasp_net, pc_keys)
    action_tensor = action.to(device, non_blocking=True)
    # action_tensor = action[:, :4].to(device, non_blocking=True)
    expected_reward_tensor = expected_reward.to(device, dtype=torch.float32, non_blocking=True)
    node_features = torch.tensor(node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edge_index, device=device)
    batch_index = torch.tensor(batch_index, dtype=torch.int64, device=device)

    predicted_reward = network(node_features, edge_index, batch_index, action_tensor)
    return predicted_reward, expected_reward_tensor


def visualize_results(predictions, actuals):
    """
    Visualize predicted vs. actual rewards.
    """
    # plt.figure(figsize=(10, 6))
    # plt.scatter(actuals, predictions, alpha=0.5, label='Predicted vs Actual')
    # plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', label='Ideal Line')
    # plt.xlabel('Actual Rewards')
    # plt.ylabel('Predicted Rewards')
    # plt.title('Predicted vs. Actual Rewards')
    # plt.legend()
    # plt.show()

    sorted_indices = np.lexsort((predictions, actuals))
    sorted_actuals = np.array(actuals)[sorted_indices]
    sorted_predictions = np.array(predictions)[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sorted_predictions)), sorted_predictions, label='Predicted Rewards', color='orange')
    plt.plot(range(len(sorted_actuals)), sorted_actuals, label='Actual Rewards', color='blue')
    plt.xlabel('Data Point Index (sorted by Actual and Predicted Rewards)')
    plt.ylabel('Rewards')
    plt.title('Predicted and Actual Rewards by Data Point Index')
    plt.legend()
    plt.savefig('sorted_predicted_vs_actual.png')
    # plt.show()


def load_model_and_data(checkpoint_path, config_path, zarr_path, batch_size=1):
    """
    Load model, data, and evaluate the network to generate plots.
    """
    # Load the configuration file
    with open(config_path, 'r') as f:
        config = json.load(f)

    graspnet_root = config['graspnet_root']
    grasp_net = GraspNet(graspnet_root, camera="kinect", split='custom')
    grasp_net_dataset = zarr.open(zarr_path, mode='r')

    # Load the dataset
    dataset = GraspDataset(grasp_net_dataset)
    val_size = len(dataset) // 10  # Use 10% for validation
    _, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    node_feature_size = 3
    action_size = 7
    # reward_network = RewardNetwork().to(device)
    reward_network = RewardNetwork(node_feature_size=node_feature_size, action_size=action_size).to(device)
    # reward_network = RewardNetwork(node_feature_size=node_feature_size, action_size=action_size, descriptor_size=64, fc_size=64).to(device)


    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    reward_network.load_state_dict(checkpoint['model_state_dict'])
    # reward_network.load_state_dict(checkpoint)
    reward_network.eval()

    # Load LRUCache
    cache = LRUCache(capacity=1000, zarr_path='cache.zarr')

    predictions = []
    actuals = []

    with torch.no_grad():
        for pc_keys, grasp_pose, expected_reward in tqdm(val_loader, desc="Evaluating"):
            predicted_reward, actual_reward = evaluate_network(reward_network, cache, grasp_net, pc_keys, grasp_pose, expected_reward)
            # predicted_reward, actual_reward = evaluate_network(reward_network, cache, grasp_net, pc_keys, grasp_pose[0], expected_reward)
            # predictions.extend(predicted_reward[0].cpu().numpy().squeeze())
            predictions.extend(predicted_reward.cpu().numpy().squeeze())
            actuals.extend(actual_reward.cpu().numpy().squeeze())

    # Visualize results
    visualize_results(predictions, actuals)


if __name__ == "__main__":
    # checkpoint_path = 'GtG_best.pth'
    checkpoint_path = 'checkpoints/reward_network_epoch_7_loss_0.0082.pth'
    # checkpoint_path = 'checkpoints_64_cos/reward_network_epoch_227_loss_0.1731.pth'
    config_path = 'pathes.json'
    zarr_path = 'grasp_net_dataset.zarr'

    load_model_and_data(checkpoint_path, config_path, zarr_path)
