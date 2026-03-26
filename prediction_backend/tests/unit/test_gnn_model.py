import torch

from prediction_backend.models.gnn_model import GNNModel
from prediction_backend.molecular_processing.graph_builder import build_graph


def test_gnn_forward():

    # Example molecule
    smiles = "CCO"

    graph = build_graph(smiles)

    # Create batch vector (single graph)
    graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)

    model = GNNModel()

    output = model(graph)

    # Output should be one prediction
    assert output.shape == (1, 1)


def test_node_feature_dimension():

    graph = build_graph("CCO")

    # Node feature size should be 7
    assert graph.x.shape[1] == 7


def test_edge_features():

    graph = build_graph("CCO")

    # Edge features should exist
    assert graph.edge_attr is not None

    # Edge feature dimension should be 3
    assert graph.edge_attr.shape[1] == 3