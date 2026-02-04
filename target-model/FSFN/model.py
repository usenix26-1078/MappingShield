# 一个GNN
# 把两个图映射到同一个特征空间
# node embedding作为mlp的输入
# 输出是否是一对的概率然后二分类

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=1, concat=False, 
                            add_self_loops=True, bias=True)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False,
                            add_self_loops=True, bias=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
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


class UNSE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=128):
        super(UNSE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 概率转移矩阵计算
        self.transition_matrix = None
        self.multi_step_matrix = None
        
        # Skip-Gram模型 - 使用input_dim作为num_embeddings
        self.embedding = nn.Embedding(input_dim, output_dim)
        self.context_embedding = nn.Embedding(input_dim, output_dim)
        
        # 初始化嵌入层
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.context_embedding.weight)
        
    def compute_transition_matrix(self, adj_matrix):
        # 确保输入在正确的设备上
        device = adj_matrix.device
        
        # 计算度矩阵
        degree_matrix = torch.sum(adj_matrix, dim=1)
        # 添加小的扰动以避免除零
        degree_matrix = degree_matrix + 1e-8
        # 计算概率转移矩阵 T^1 = D^(-1)A
        # 使用更稳定的方法：逐行归一化
        self.transition_matrix = adj_matrix / degree_matrix.unsqueeze(1)
        
    def compute_multi_step_matrix(self, k=3):
        # 确保在正确的设备上
        device = self.transition_matrix.device
        
        # 计算多步转移矩阵 M = T^1 + T^2 + ... + T^k
        self.multi_step_matrix = self.transition_matrix
        current_matrix = self.transition_matrix
        
        for i in range(2, k+1):
            # 使用矩阵乘法而不是幂运算
            current_matrix = torch.mm(current_matrix, self.transition_matrix)
            self.multi_step_matrix = self.multi_step_matrix + current_matrix
            
    def compute_common_neighbors(self, node_i, node_j):
        # 确保在正确的设备上
        device = self.multi_step_matrix.device
        node_i = node_i.to(device)
        node_j = node_j.to(device)
        
        # 计算共同邻居向量 CN_i
        cn_i = torch.sum(self.multi_step_matrix[node_i] * self.multi_step_matrix[node_j], dim=1)
        # 计算存在向量 exCN_i
        ex_cn_i = torch.sum(self.multi_step_matrix[node_i] * self.multi_step_matrix[node_j], dim=0)
        return cn_i, ex_cn_i
    
    def forward(self, node_ids):
        # 确保节点索引在正确的设备上
        device = node_ids.device
        
        # 确保节点索引在有效范围内
        max_node_id = node_ids.max().item()
        if max_node_id >= self.input_dim:
            # 动态扩展嵌入层
            old_embedding = self.embedding
            old_context_embedding = self.context_embedding
            
            self.input_dim = max_node_id + 1
            self.embedding = nn.Embedding(self.input_dim, self.output_dim).to(device)
            self.context_embedding = nn.Embedding(self.input_dim, self.output_dim).to(device)
            
            # 复制旧的权重
            self.embedding.weight.data[:old_embedding.weight.size(0)] = old_embedding.weight.data
            self.context_embedding.weight.data[:old_context_embedding.weight.size(0)] = old_context_embedding.weight.data
            
            # 初始化新的权重
            nn.init.xavier_uniform_(self.embedding.weight.data[old_embedding.weight.size(0):])
            nn.init.xavier_uniform_(self.context_embedding.weight.data[old_context_embedding.weight.size(0):])
        
        # 生成节点嵌入
        node_embeddings = self.embedding(node_ids)
        return node_embeddings

class AnchorComponent(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(AnchorComponent, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.encoder(x)

class FusionComponent(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(FusionComponent, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.discriminator(x)

class AMFF(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(AMFF, self).__init__()
        self.anchor_component = AnchorComponent(input_dim, hidden_dim)
        self.fusion_component = FusionComponent(input_dim, hidden_dim)
        self.tau = 1.0  # 锚损失权重
        self.rho = 0.25  # 融合损失权重
        self.eta = 0.1  # 融合权重
        
    def forward(self, source_emb, target_emb):
        # 锚组件处理
        source_anchor = self.anchor_component(source_emb)
        target_anchor = self.anchor_component(target_emb)
        
        # 融合组件处理
        source_fusion = self.fusion_component(source_emb)
        target_fusion = self.fusion_component(target_emb)
        
        # 特征融合
        fused_source = source_emb + self.eta * source_fusion
        fused_target = target_emb + self.eta * target_fusion
        
        # 标准化
        fused_source = F.normalize(fused_source, p=2, dim=1)
        fused_target = F.normalize(fused_target, p=2, dim=1)
        
        return fused_source, fused_target

class CrossNetworkClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dims=[256, 128, 64]):
        super(CrossNetworkClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Sigmoid(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Sigmoid(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.Sigmoid(),
            nn.Linear(hidden_dims[2], 1),
            nn.Sigmoid()
        )
        
    def forward(self, source_emb, target_emb):
        # 连接源网络和目标网络的嵌入
        combined_emb = torch.cat([source_emb, target_emb], dim=1)
        return self.classifier(combined_emb)

class UNSE_AMFF(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=128):
        super(UNSE_AMFF, self).__init__()
        self.unse = UNSE(input_dim, hidden_dim, output_dim)
        self.amff = AMFF(output_dim, hidden_dim)
        self.classifier = CrossNetworkClassifier(output_dim * 2)
        
    def forward(self, source_adj, target_adj, source_nodes, target_nodes):
        # UNSE处理
        self.unse.compute_transition_matrix(source_adj)
        self.unse.compute_multi_step_matrix()
        source_emb = self.unse(source_nodes)
        
        self.unse.compute_transition_matrix(target_adj)
        self.unse.compute_multi_step_matrix()
        target_emb = self.unse(target_nodes)
        
        # AMFF处理
        fused_source, fused_target = self.amff(source_emb, target_emb)
        
        # 分类
        pred = self.classifier(fused_source, fused_target)
        return pred

