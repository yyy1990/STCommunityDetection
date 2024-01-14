import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class GGT_layer(MessagePassing):

    def __init__(self, in_channels, out_channels, device, heads=1, dropout=0., edge_dim=None, res_type='gated', attention_type='joint',**kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.res_type = res_type
        self.attention_type = attention_type

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self._alpha = None

        self.key_linear = nn.Linear(in_channels, heads * out_channels)
        self.query_linear = nn.Linear(in_channels, heads * out_channels)
        self.value_linear = nn.Linear(in_channels, heads * out_channels)
        self.edge_linear = nn.Linear(edge_dim, heads * out_channels)

        self.skip_linear = nn.Linear(in_channels, heads * out_channels)

        self.sigmoid = nn.Sigmoid()
        self.ffn = nn.Linear(heads * out_channels, heads * out_channels)

        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(heads * out_channels)

        self.gate = nn.Linear(4*heads * out_channels, 1)

        self.node_key_weight = torch.tensor(1.0, requires_grad=True, device=device)
        self.edge_key_weight = torch.tensor(1.0, requires_grad=True, device=device)

    def forward(self, x, edge_index, edge_attr):
        heads, channels = self.heads, self.out_channels

        query = self.query_linear(x).view(-1, heads, channels)
        key = self.key_linear(x).view(-1, heads, channels)
        value = self.value_linear(x).view(-1, heads, channels)

        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr)

        self._alpha = None

        out = out.view(-1, self.heads * self.out_channels)

        x_skip = self.skip_linear(x)

        attention_out = out+x_skip
        attention_out = self.layer_norm(attention_out)
        attention_out = self.gelu(attention_out)

        ffn_out = self.ffn(attention_out)
        ffn_out = out+ffn_out
        ffn_out = self.layer_norm(ffn_out)
        ffn_out = self.gelu(ffn_out)

        gate_coef = self.gate(torch.cat([x_skip, ffn_out, ffn_out - x_skip, ffn_out*x_skip], dim=-1))
        gate_coef = self.sigmoid(gate_coef)

        if self.res_type=='gated':
            res = gate_coef * x_skip + (1 - gate_coef) * ffn_out
        elif self.res_type=='add':
            res = x_skip + ffn_out
        else:
            res = ffn_out

        return res

    def message(self, query_i, key_j, value_j, edge_attr, index, ptr):

        edge_attr = self.edge_linear(edge_attr).view(-1, self.heads, self.out_channels)
        if self.attention_type=='joint':
            key_j = self.node_key_weight*key_j + self.edge_key_weight*edge_attr
        elif self.attention_type=='node':
            key_j = self.node_key_weight*key_j
        elif self.attention_type=='edge':
            key_j = self.edge_key_weight*edge_attr

        attention_weight = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        attention_weight = softmax(attention_weight, index, ptr)
        self._alpha = attention_weight
        attention_weight = F.dropout(attention_weight, p=self.dropout, training=self.training)

        out = value_j
        out = out + edge_attr

        out = out * attention_weight.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, dropout={self.dropout})')
