from torch_geometric.nn import GCNConv

from .base import (OneGraphConvolution, ThreeGraphConvolution,
                   TwoGraphConvolution)


class ThreeGCN(ThreeGraphConvolution):
    def _get_convolution_module(self, in_channels: int, out_channels: int) -> GCNConv:
        return GCNConv(in_channels=in_channels, out_channels=out_channels)


class TwoGCN(TwoGraphConvolution):
    def _get_convolution_module(self, in_channels: int, out_channels: int) -> GCNConv:
        return GCNConv(in_channels=in_channels, out_channels=out_channels)


class OneGCN(OneGraphConvolution):
    def _get_convolution_module(self, in_channels: int, out_channels: int) -> GCNConv:
        return GCNConv(in_channels=in_channels, out_channels=out_channels)
