from torch_geometric.nn import GATv2Conv

from .base import (OneGraphConvolution, ThreeGraphConvolution,
                   TwoGraphConvolution)


class ThreeGATCN(ThreeGraphConvolution):
    def _get_convolution_module(self, in_channels: int, out_channels: int) -> GATv2Conv:
        return GATv2Conv(in_channels=in_channels, out_channels=out_channels)


class TwoGATCN(TwoGraphConvolution):
    def _get_convolution_module(self, in_channels: int, out_channels: int) -> GATv2Conv:
        return GATv2Conv(in_channels=in_channels, out_channels=out_channels)


class OneGATCN(OneGraphConvolution):
    def _get_convolution_module(self, in_channels: int, out_channels: int) -> GATv2Conv:
        return GATv2Conv(in_channels=in_channels, out_channels=out_channels)
