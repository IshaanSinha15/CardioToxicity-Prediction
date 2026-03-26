import torch
import torch.nn as nn

class FusionSingleTask(nn.Module):

    def __init__(self):
        super().__init__()

        # Smaller projections (reduce capacity)
        self.chem_proj = nn.Linear(768, 128)
        self.gnn_proj = nn.Linear(128, 128)

        # Simple fusion (NO attention)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)

        self.out = nn.Linear(64, 1)

        self.relu = nn.ReLU()

        # Strong regularization
        self.dropout = nn.Dropout(0.4)

    def forward(self, chem_emb, gnn_emb):

        chem = self.chem_proj(chem_emb)
        gnn  = self.gnn_proj(gnn_emb)

        x = torch.cat([chem, gnn], dim=1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))

        return self.out(x)