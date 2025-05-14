import zarr
import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from gs3d.utils.lru_cache import LRUCache
from gs3d.RL.model import RewardNetwork, GraspDataset
from gs3d.utils.model_utils import pc_to_graph_from_zarr

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_checkpoint(checkpoint_path, model):
    """
    Load model and optimizer state from a checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Checkpoint loaded: Epoch {epoch}, Loss {loss:.4f}")
    return model


def evaluate_model(model, data_loader, cache):
    """
    Evaluate the trained model on the test data and visualize the results.
    """
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_actuals = []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for pc_keys, grasp_pose, grasp_points, expected_reward in tqdm(data_loader, desc='Evaluating'):
            # Convert point cloud data to graph format
            node_features, edge_index, batch_index = pc_to_graph_from_zarr(
                data_loader.dataset.zarr_store,
                cache,
                pc_keys
            )

            # Prepare tensors for model input
            action_tensor = grasp_pose.to(device)
            expected_reward_tensor = expected_reward.to(device, dtype=torch.float32)
            node_features = torch.tensor(node_features, dtype=torch.float32).to(device)
            edge_index = torch.tensor(edge_index).to(device)
            batch_index = torch.tensor(batch_index, dtype=torch.int64).to(device)

            # Run the model to get predictions
            predicted_reward = model(node_features, edge_index, batch_index, action_tensor)

            # Collect predictions and actuals
            all_predictions.extend(predicted_reward.cpu().numpy())
            all_actuals.extend(expected_reward_tensor.cpu().numpy())

    return all_predictions, all_actuals


def visualize_results(predictions, actuals):
    """
    Visualize predicted vs. actual rewards.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5, label='Predicted vs Actual')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', label='Ideal Line')
    plt.xlabel('Actual Rewards')
    plt.ylabel('Predicted Rewards')
    plt.title('Predicted vs. Actual Rewards')
    plt.legend()
    plt.show()

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
    plt.show()


def main():
    # Load the dataset
    cache = LRUCache(capacity=1000, zarr_path='cache.zarr')
    zarr_store = zarr.open('dataset.zarr', mode='r')
    dataset = GraspDataset(zarr_store)
    data_loader = DataLoader(dataset, batch_size=100, shuffle=False)

    # Initialize the model and optimizer
    node_feature_size = 3  # Assuming point cloud has 3 features (x, y, z)
    action_size = 7  # Assuming action has 7 features
    model = RewardNetwork(node_feature_size=node_feature_size, action_size=action_size, descriptor_size=32, fc_size=32).to(device)

    # Load the checkpoint
    # checkpoint_path = 'checkpoints_64_cos/reward_network_epoch_232_loss_0.1827.pth'
    checkpoint_path = 'checkpoints/reward_network_epoch_189_loss_0.2669.pth'
    model = load_checkpoint(checkpoint_path, model)

    # Evaluate the model
    print("Evaluating the model...")
    predictions, actuals = evaluate_model(model, data_loader, cache)

    # Visualize the results
    visualize_results(predictions, actuals)


if __name__ == "__main__":
    main()
