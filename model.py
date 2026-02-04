# 一个GNN
# 把两个图映射到同一个特征空间
# node embedding作为mlp的输入
# 输出是否是一对的概率然后二分类

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool



class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATEncoder, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels // 8, heads=8, dropout=0.6)
        self.gat2 = GATConv(hidden_channels, out_channels // 8, heads=8, dropout=0.6)
    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return x

class GraphSAGEEncoder(nn.Module):
    """无参数均值聚合（直接均值）"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
    def forward(self, x, edge_index):
        for _ in range(2):
            row, col = edge_index
            agg = torch.zeros_like(x)
            agg.index_add_(0, row, x[col])
            deg = torch.zeros(x.size(0), device=x.device)
            deg.index_add_(0, row, torch.ones_like(row, dtype=x.dtype))
            deg = deg.clamp(min=1).unsqueeze(-1)
            x = (x + agg) / (deg + 1)
        return x

class GraphSAGEEncoderMeanPool(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEEncoderMeanPool, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GraphSAGEEncoderMaxPool(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEEncoderMaxPool, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='max')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='max')
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
    def forward(self, embedding1, embedding2):
        x = torch.cat([embedding1, embedding2], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.out(x))
        return x


# 修改后的generator定义
class generator(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, in_features)
        self.global_perturbation = nn.Parameter(torch.randn(1, in_features))  # 全局可学习的扰动

    def forward(self, node_feature, alpha):
        x = alpha * node_feature + (1 - alpha) * self.global_perturbation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

