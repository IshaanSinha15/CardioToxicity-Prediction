import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import GINEConv, global_mean_pool


class GNNModel(torch.nn.Module):

    def __init__(self, input_dim=7, hidden_dim=128):

        super(GNNModel, self).__init__()

        # ---- Layer 1 ----
        nn1 = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINEConv(nn1, edge_dim=3)
        self.bn1 = BatchNorm1d(hidden_dim)

        # ---- Layer 2 ----
        nn2 = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINEConv(nn2, edge_dim=3)
        self.bn2 = BatchNorm1d(hidden_dim)

        # ---- Layer 3 ----
        nn3 = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.conv3 = GINEConv(nn3, edge_dim=3)
        self.bn3 = BatchNorm1d(hidden_dim)

        # regularization
        self.dropout = Dropout(0.3)

        # MLP head
        self.fc1 = Linear(hidden_dim, 64)
        self.fc2 = Linear(64, 1)

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        # GNN Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        # GNN Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        # GNN Layer 3
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        # Graph pooling
        x = global_mean_pool(x, batch)

        # Dense layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x