import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool
import torch.nn.functional as F


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

    def forward(self, x, edge_index, action):
        hidden0 = self.initial_conv(x, edge_index)

        hidden1 = self._gcn_block(hidden0, self.conv1, edge_index)
        hidden2 = self._gcn_block(hidden1, self.conv2, edge_index)
        hidden3 = self._gcn_block(hidden2, self.conv3, edge_index)

        x = global_max_pool(hidden3)
        V = self.ln2(x)

        action = self._action_block(action)

        x = V * action
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.ln4(x)
        x = self.fc2(x)

        return x.squeeze(1)