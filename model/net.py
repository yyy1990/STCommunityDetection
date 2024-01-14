import torch
from ggt import GGT_layer

class GraphTransformer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads, dropout, device):
        super(GraphTransformer, self).__init__()
        self.layer1 = GGT_layer(in_channels=input_dim, out_channels=output_dim, edge_dim=1, heads=heads, dropout=dropout, beta=True, device=device)
        self.layer2to8 = GGT_layer(in_channels=output_dim*heads, out_channels=output_dim, edge_dim=1, heads=heads, dropout=dropout, beta=True, device=device)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            edge_weight = torch.unsqueeze(edge_weight, dim=-1)
        x = self.layer1(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x1 = self.layer2to8(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x1 = self.layer2to8(x=x1, edge_index=edge_index, edge_attr=edge_weight)
        x1 = self.layer2to8(x=x1, edge_index=edge_index, edge_attr=edge_weight)
        x1 = self.layer2to8(x=x1, edge_index=edge_index, edge_attr=edge_weight)
        x1 = self.layer2to8(x=x1, edge_index=edge_index, edge_attr=edge_weight)
        x1 = self.layer2to8(x=x1, edge_index=edge_index, edge_attr=edge_weight)
        x1 = self.layer2to8(x=x1, edge_index=edge_index, edge_attr=edge_weight)
        return x1
