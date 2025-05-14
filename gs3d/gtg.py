import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool

node_feature_size = 3  # Dimension of node features
embedding_size = 128

descriptor_size = 128  # Dimension of the descriptor output
action_size = 5  # Dimension of the action input
hidden_size = 128  # Dimension of hidden layers in the Rmlp subnet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RewardNetwork(nn.Module):
    def __init__(self):
        super(RewardNetwork, self).__init__()
        self.initial_conv = GCNConv(node_feature_size, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # self.do = nn.Dropout(0.5)

        self.ln1 = nn.LayerNorm(descriptor_size)  # Layer normalization
        self.ln2 = nn.LayerNorm(descriptor_size)
        # self.ln3 = nn.LayerNorm(descriptor_size)

        self.afc1 = nn.Linear(action_size, 8 * action_size)
        self.afc2 = nn.Linear(8 * action_size, 8 * action_size)
        self.afc3 = nn.Linear(8 * action_size, descriptor_size)

        self.fc1 = nn.Linear(descriptor_size, 64)
        self.fc2 = nn.Linear(64, 1)
        # self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, 1)  # Output layer

        # Xavier initialization for better stability
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.afc1.weight)
        # nn.init.xavier_uniform_(self.afc1.bias)
        nn.init.xavier_uniform_(self.afc2.weight)
        # nn.init.xavier_uniform_(self.afc2.bias)
        nn.init.xavier_uniform_(self.afc3.weight)
        # nn.init.xavier_uniform_(self.afc3.bias)

        # Layer normalization
        # self.ln_2 = nn.LayerNorm(action_size)
        # self.ln_2 = nn.LayerNorm(4*action_size)
        self.ln_3 = nn.LayerNorm(4 * action_size)
        self.ln_4 = nn.LayerNorm(64)
        self.ln_1 = nn.LayerNorm(8 * action_size)

    def forward(self, x, edge_index, batch_index, a):
        hidden0 = self.initial_conv(x, edge_index)

        hidden1 = self.conv1(hidden0, edge_index)
        hidden1 = F.leaky_relu(hidden1)  # Apply layer normalization
        hidden1 = hidden1 + hidden0  # Skip connection

        hidden2 = self.conv2(hidden1, edge_index)
        hidden2 = F.leaky_relu(hidden2)
        hidden2 = hidden2 + hidden1

        hidden3 = self.conv3(hidden2, edge_index)
        hidden3 = F.leaky_relu(hidden3)
        hidden3 = hidden3 + hidden2

        x = global_max_pool(hidden3, batch_index)
        # x = self.ln_1(x)

        V = x.flatten().unsqueeze(0)

        # Combine descriptor and action
        V = V.repeat(a.size(0), 1)
        V = self.ln2(V)

        sin_theta = (torch.sin(a[:, 3]) / 20).reshape(-1, 1)
        cos_theta = (torch.cos(a[:, 3]) / 20).reshape(-1, 1)
        a = torch.cat([a[:, 0].reshape(-1, 1), a[:, 1].reshape(-1, 1), a[:, 2].reshape(-1, 1), sin_theta, cos_theta],
                      dim=1).to(device).reshape(-1, 5)

        # a=self.ln_2(a)
        a = self.afc1(a)
        a = F.leaky_relu(a)
        a = self.ln_1(a)

        a = self.afc2(a)
        a = F.leaky_relu(a)
        # a = self.ln_2(a)

        a = self.afc3(a)
        a = F.leaky_relu(a)
        a = self.ln1(a)

        x = V * a

        # x = self.do(x)

        # x = self.ln3(x)

        # Pass through layers with activation functions and layer normalization

        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.ln_4(x)

        x = self.fc2(x)
        # x = F.sigmoid(x)

        # Output layer
        reward = x

        return reward, V


# GtG = RewardNetwork().to(device)
# print("GtG Params: " + str(sum(p.numel() for p in GtG.parameters())))
# GtG.load_state_dict(torch.load('GtG_best.pth'))
# GtG.eval()
