from typing import Any

import torch
import torch.nn
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing


class GraphConvolutionBase(torch.nn.Module):
    def _get_convolution_module(
        self, in_channels: int, out_channels: int
    ) -> MessagePassing:
        raise NotImplementedError()


class ThreeGraphConvolution(GraphConvolutionBase):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = self._get_convolution_module(num_features, 1024)
        self.conv2 = self._get_convolution_module(1024, 512)
        self.conv3 = self._get_convolution_module(512, 128)
        self.fc = Linear(128, num_classes)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)
        h = h.relu()  # Final GNN embedding space.
        # Apply a final (linear) classifier.
        return self.fc(h)


class TwoGraphConvolution(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = self._get_convolution_module(num_features, 256)
        self.conv2 = self._get_convolution_module(256, 128)
        self.fc = Linear(128, num_classes)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        # Apply a final (linear) classifier.
        return self.fc(h)


class OneGraphConvolution(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = self._get_convolution_module(num_features, 1024)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 128)
        self.classifier = Linear(128, num_classes)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.fc1(h)
        h = h.relu()
        h = self.fc2(h)
        h = h.relu()
        # Apply a final (linear) classifier.
        return self.classifier(h)
