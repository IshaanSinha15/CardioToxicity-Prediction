import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm

from torch_geometric.nn import GINEConv, global_mean_pool


class GNNModel(torch.nn.Module):

    def __init__(self, node_dim=7, edge_dim=3, hidden_dim=128):

        super(GNNModel, self).__init__()

        self.input_proj = Linear(node_dim, hidden_dim)

        # ----- Message Passing Layers -----

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for _ in range(4):

            nn = torch.nn.Sequential(
                Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                Linear(hidden_dim, hidden_dim)
            )

            self.convs.append(GINEConv(nn, edge_dim=edge_dim))
            self.norms.append(LayerNorm(hidden_dim))

        # ----- Readout head -----

        self.dropout = Dropout(0.2)

        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = Linear(hidden_dim // 2, 1)

    def forward(self, data, return_embedding=False):

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )

        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):

            h = conv(x, edge_index, edge_attr)
            h = F.relu(h)

            x = x + h
            x = norm(x)

        # ----- Pooling -----
        x = global_mean_pool(x, batch)

        # return embedding for fusion
        if return_embedding:
            return x

        # ----- MLP head -----
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x