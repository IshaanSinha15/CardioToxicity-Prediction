import torch
from prediction_backend.models.gnn_model import GNNModel
from torch_geometric.nn import global_mean_pool


class GNNEncoder:

    def __init__(self, model_path, device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GNNModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()

    def encode(self, batch):

        batch = batch.to(self.device)

        with torch.no_grad():

            x, edge_index, edge_attr, batch_idx = (
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )

            # SAME LOGIC AS TRAINED GNN (IMPORTANT)
            x = self.model.input_proj(x)

            for conv, norm in zip(self.model.convs, self.model.norms):

                h = conv(x, edge_index, edge_attr)
                h = torch.relu(h)

                x = x + h
                x = norm(x)

            # graph-level embedding
            graph_emb = global_mean_pool(x, batch_idx)

        return graph_emb  # [batch_size, 128]