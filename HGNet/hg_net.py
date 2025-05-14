import torch
import torch.nn as nn
from knn_pytorch.knn_pytorch import knn
from torch_geometric.nn import DynamicEdgeConv


class GravNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GravNetBlock, self).__init__()
        # Pre-processing layers
        self.pre_layers = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # GravNetConv
        self.conv = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * 64, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            ),
            k=16
        )

        # Post-processing layers
        self.post_layers = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

        # Skip connection layer
        self.skip_lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, batch):
        x = self.pre_layers(x)
        x = self.conv(x, batch=batch)
        x = self.post_layers(x)
        return x


class HGGQNet(nn.Module):
    def __init__(self,
                 graph_feature_dim=3,
                 graph_hidden_dim=256,
                 graph_out_dim=128,
                 num_gnn_layers=3):
        super(HGGQNet, self).__init__()

        self.center_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.direction_encoder = nn.Sequential(
            nn.Linear(9, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.depth_encoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU()
        )
        self.width_encoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU()
        )

        combined_input_dim = (self.center_encoder[-3].out_features
                              + self.direction_encoder[-3].out_features
                              + self.depth_encoder[-3].out_features
                              + self.width_encoder[-3].out_features)

        # Encoder-Decoder for grasp features
        latent_dim = 32
        self.grasp_encoder = nn.Sequential(
            nn.Linear(combined_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

        self.grasp_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, combined_input_dim),
            nn.BatchNorm1d(combined_input_dim),
            nn.ReLU()
        )

        self.graph_hidden_dim = graph_hidden_dim
        self.graph_out_dim = graph_out_dim

        self.gnn_blocks = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_dim = graph_feature_dim if i == 0 else graph_hidden_dim
            out_dim = graph_hidden_dim if i < num_gnn_layers - 1 else graph_out_dim
            self.gnn_blocks.append(GravNetBlock(in_dim, out_dim))

        combined_feat_dim = (graph_out_dim + combined_input_dim)

        # Score prediction from combined features
        self.score_predictor = nn.Sequential(
            nn.Linear(combined_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def process_grasp(self, grasp):
        """Encode and then decode the grasp configuration into a fixed-length feature."""
        # Extract and encode individual components
        center_feat = self.center_encoder(grasp[:, :3])          # (N,16)
        direction_feat = self.direction_encoder(grasp[:, 3:12])  # (N,16)
        depth_feat = self.depth_encoder(grasp[:, 12:13])         # (N,4)
        width_feat = self.width_encoder(grasp[:, 13:14])         # (N,4)

        # Combine features
        combined_feat = torch.cat([center_feat, direction_feat, depth_feat, width_feat], dim=-1)

        # Encode into latent space and then decode back
        latent = self.grasp_encoder(combined_feat)   # (N, latent_dim)
        decoded_feat = self.grasp_decoder(latent)    # (N, combined_input_dim)

        return decoded_feat

    def graph_forward(self, graph):
        """Pass the graph through GNN layers."""
        # Concatenate xyz features
        node_feats = graph.xyz  # [N_nodes, 3]

        # Pass through each GravNet block
        for block in self.gnn_blocks:
            node_feats = block(node_feats, batch=graph.batch)

        graph.graph_feat = node_feats

    def aggregate_features(self, graph, grasp_feat, grasp_centers, graph_indices):
        """Aggregate graph features with grasp configuration features using KNN with k=1."""
        # Get node positions and features
        node_positions = graph.xyz  # Shape: [N_nodes, 3]
        node_features = graph.graph_feat  # Shape: [N_nodes, F]

        batch_num_nodes = torch.bincount(graph.xyz_batch, minlength=graph.num_graphs)

        # Create a tensor indicating the graph index for each node
        node_graph_indices = torch.repeat_interleave(
            torch.arange(len(batch_num_nodes), device=node_positions.device),
            batch_num_nodes
        ).to(node_positions.device)

        # Initialize tensor to store aggregated features
        aggregated_features = torch.zeros(grasp_centers.size(0), node_features.size(1), device=node_positions.device)

        # Get unique graph indices
        unique_graph_indices = torch.unique(graph_indices)

        for graph_index in unique_graph_indices:
            # Filter nodes and grasps for the current graph
            grasp_mask = (graph_indices == graph_index)
            node_mask = (node_graph_indices == graph_index)

            grasps_in_graph = grasp_centers[grasp_mask]       # [G, 3]
            nodes_in_graph = node_positions[node_mask]        # [N, 3]
            features_in_graph = node_features[node_mask]      # [N, F]

            max_chunk_size = 200000  # Adjust based on memory constraints
            num_grasps = grasps_in_graph.size(0)
            chunk_size = max_chunk_size if num_grasps > max_chunk_size else num_grasps

            for start_idx in range(0, num_grasps, chunk_size):
                end_idx = min(start_idx + chunk_size, num_grasps)
                grasps_chunk = grasps_in_graph[start_idx:end_idx]  # [C, 3]

                # Prepare data for KNN
                reference_points = nodes_in_graph  # [N, 3]
                query_points = grasps_chunk        # [C, 3]

                # Transpose and reshape for KNN function
                reference_points_t = reference_points.permute(1, 0).contiguous().unsqueeze(0)  # [1, 3, N]
                query_points_t = query_points.permute(1, 0).contiguous().unsqueeze(0)          # [1, 3, C]

                k = 1
                knn_indices = torch.empty(1, k, query_points.size(0), dtype=torch.long, device=reference_points.device)

                knn(reference_points_t, query_points_t, knn_indices)

                knn_indices = knn_indices.squeeze(0).squeeze(0) - 1  # Shape: [C]
                knn_indices = torch.clamp(knn_indices, 0, nodes_in_graph.size(0) - 1)

                # Gather features from the nearest nodes
                grasp_features_chunk = features_in_graph[knn_indices]  # [C, F]

                # Store the aggregated features
                aggregated_features[grasp_mask][start_idx:end_idx] = grasp_features_chunk

        # Combine aggregated graph features and grasp configuration features
        return torch.cat([grasp_feat, aggregated_features], dim=-1)

    def forward(self, grasp_config, graph, graph_indices):
        # Encode and decode grasp configuration
        grasp_features = self.process_grasp(grasp_config)

        # Pass the graph through the GNN
        self.graph_forward(graph)

        # Aggregate graph and grasp features
        grasp_centers = grasp_config[:, :3]  # Use grasp center for aggregation
        combined_features = self.aggregate_features(graph, grasp_features, grasp_centers, graph_indices)

        # Predict grasp score
        scores = self.score_predictor(combined_features).squeeze(-1)
        center_scores = graph.graph_feat[:, 0]
        fingers_scores = graph.graph_feat[:, 1]

        return scores, center_scores, fingers_scores
