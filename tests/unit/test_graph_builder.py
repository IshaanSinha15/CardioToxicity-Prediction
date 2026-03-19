import pytest
from torch_geometric.data import Data
from molecular_processing.graph_builder import build_graph


def test_graph_creation():
    graph = build_graph("CCO")

    assert isinstance(graph, Data)
    assert graph.x.shape[0] > 0      # nodes
    assert graph.x.shape[1] > 0      # features
    assert graph.edge_index.shape[1] > 0  # edges


def test_invalid_smiles():
    with pytest.raises(ValueError):
        build_graph("INVALID")