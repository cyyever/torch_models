import torch.nn
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv


class ThreeGATCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, 1024)
        self.conv2 = GATv2Conv(1024, 512)
        self.conv3 = GATv2Conv(512, 128)
        self.classifier = Linear(128, num_classes)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)
        h = h.relu()  # Final GNN embedding space.
        # Apply a final (linear) classifier.
        return self.classifier(h)


class TwoGATCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, 256)
        self.conv2 = GATv2Conv(256, 128)
        self.classifier = Linear(128, num_classes)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        # Apply a final (linear) classifier.
        return self.classifier(h)


class OneGATCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, 1024)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 128)
        self.classifier = Linear(128, num_classes)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.fc1(h)
        h = h.relu()
        h = self.fc2(h)
        h = h.relu()
        # Apply a final (linear) classifier.
        return self.classifier(h)
