# 一个GNN
# 把两个图映射到同一个特征空间
# node embedding作为mlp的输入
# 输出是否是一对的概率然后二分类

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # 添加注意力机制
        self.attention1 = nn.Linear(hidden_channels, 1)
        self.attention2 = nn.Linear(out_channels, 1)
        
        # 初始化权重
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        nn.init.xavier_uniform_(self.attention1.weight)
        nn.init.xavier_uniform_(self.attention2.weight)
        nn.init.zeros_(self.attention1.bias)
        nn.init.zeros_(self.attention2.bias)

    def forward(self, x, edge_index):
        # 确保输入在正确的设备上
        x = x.to(edge_index.device)
        
        # 第一层卷积
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # 应用注意力
        att1 = torch.sigmoid(self.attention1(x))
        x = x * att1
        
        # 第二层卷积
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        
        # 应用注意力
        att2 = torch.sigmoid(self.attention2(x))
        x = x * att2
        
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, 1)
        self.dropout = nn.Dropout(0.2)
        
        # 初始化权重
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, embedding1, embedding2):
        x = torch.cat([embedding1, embedding2], dim=-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# 修改后的generator定义
class generator(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, in_features)
        self.global_perturbation = nn.Parameter(torch.randn(1, in_features))  # 全局可学习的扰动
        
        # 初始化权重
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, node_feature, alpha):
        x = alpha * node_feature + (1 - alpha) * self.global_perturbation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

