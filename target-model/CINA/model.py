# 一个GNN
# 把两个图映射到同一个特征空间
# node embedding作为mlp的输入
# 输出是否是一对的概率然后二分类

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class HypergraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HypergraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        # 使用scatter操作来实现超图卷积
        num_nodes = x.size(0)
        
        # 计算每个节点的邻居
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg = deg.clamp(min=1)  # 避免除零
        
        # 计算归一化权重
        norm = 1.0 / deg[row]
        
        # 聚合邻居信息
        out = torch.zeros(num_nodes, self.in_channels, device=x.device)
        out.scatter_add_(0, row.unsqueeze(-1).expand(-1, self.in_channels), 
                        x[col] * norm.unsqueeze(-1))
        
        # 线性变换
        out = out @ self.weight + self.bias
        return out

class CINA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CINA, self).__init__()
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, hidden_channels)
        self.conv3 = HypergraphConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, 1)
        # self.fc3 = nn.Linear(in_features,1)

    def forward(self, embedding1, embedding2):
        x = torch.cat([embedding1, embedding2], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
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

