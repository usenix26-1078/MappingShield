import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import random

class GCNEmbedder(nn.Module):
    """GCN嵌入模型"""
    def __init__(self, in_channels, hidden_channels=128):
        super(GCNEmbedder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, in_channels)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def train_embedder(model, data, device):
    """训练嵌入模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.mse_loss(out, data.x.to(device))
        loss.backward()
        optimizer.step()
    
    return model

def dimension_reduction(features, method='tsne'):
    """降维处理
    Args:
        features: 输入特征
        method: 降维方法，这里固定使用t-SNE
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    
    print(f"原始特征形状: {features.shape}")
    print(f"特征值范围: [{np.min(features)}, {np.max(features)}]")
    
    # 简单的数据清理
    features = np.nan_to_num(features, nan=0.0)
    features = np.clip(features, -1e10, 1e10)
    
    # 标准化
    features_std = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    
    try:
        print("使用t-SNE进行降维...")
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(features_std)
        
        print(f"降维后特征形状: {reduced_features.shape}")
        print(f"降维后特征值范围: [{np.min(reduced_features)}, {np.max(reduced_features)}]")
        return reduced_features
        
    except Exception as e:
        print(f"降维过程出错: {str(e)}")
        print("特征统计信息:")
        print(f"Shape: {features.shape}")
        print(f"Mean: {np.mean(features)}")
        print(f"Std: {np.std(features)}")
        print(f"Min: {np.min(features)}")
        print(f"Max: {np.max(features)}")
        print(f"NaN count: {np.isnan(features).sum()}")
        print(f"Inf count: {np.isinf(features).sum()}")
        raise e

def visualize_embeddings(reduced_features, normal_nodes, target_nodes, trigger_nodes, 
                        title, save_path, filename):
    """可视化节点分布"""
    plt.figure(figsize=(10, 8))
    
    # 绘制普通节点
    plt.scatter(reduced_features[list(normal_nodes), 0], 
               reduced_features[list(normal_nodes), 1],
               c='green', alpha=0.6, label='Normal Nodes')
    
    # 绘制目标节点
    plt.scatter(reduced_features[list(target_nodes), 0],
               reduced_features[list(target_nodes), 1],
               c='blue', marker='^', s=100, label='Target Nodes')
    
    # 绘制触发器节点
    plt.scatter(reduced_features[list(trigger_nodes), 0],
               reduced_features[list(trigger_nodes), 1],
               c='red', marker='*', s=100, label='Trigger Nodes')
    
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_node_distribution(G, target_nodes, trigger_nodes, save_path, graph_name, use_embedding=True):
    """分析节点分布"""
    # 强制使用CPU
    device = torch.device('cpu')
    
    print(f"\n开始分析 {graph_name} 的节点分布...")
    print(f"图节点数量: {G.x.size(0)}")
    print(f"特征维度: {G.x.size(1)}")
    
    # 获取节点特征
    if use_embedding:
        print("使用GCN进行特征嵌入...")
        embedder = GCNEmbedder(G.x.size(1))
        embedder = train_embedder(embedder, G, device)
        with torch.no_grad():
            features = embedder(G.x.to(device), G.edge_index.to(device))
    else:
        print("使用原始特征...")
        features = G.x.to(device)
    
    print(f"特征张量形状: {features.shape}")
    
    # 检查特征值
    if torch.isnan(features).any() or torch.isinf(features).any():
        print(f"警告：{graph_name} 的特征中存在NaN或inf值")
        print(f"NaN数量: {torch.isnan(features).sum().item()}")
        print(f"Inf数量: {torch.isinf(features).sum().item()}")
        features = torch.nan_to_num(features, nan=0.0)
        features = torch.clamp(features, min=-1e10, max=1e10)
    
    # 获取各类节点索引
    target_node_indices = set(target_nodes.keys())
    trigger_node_indices = set()
    for triggers in trigger_nodes.values():
        trigger_node_indices.update(triggers.tolist())
    normal_node_indices = set(range(G.x.size(0))) - target_node_indices - trigger_node_indices
    
    # 如果普通节点太多，随机采样
    if len(normal_node_indices) > 500:
        normal_node_indices = set(random.sample(list(normal_node_indices), 500))
    
    print(f"目标节点数量: {len(target_node_indices)}")
    print(f"触发器节点数量: {len(trigger_node_indices)}")
    print(f"普通节点数量: {len(normal_node_indices)}")
    
    try:
        print("\n开始降维处理...")
        reduced_features = dimension_reduction(features, method='tsne')
        title = f'{graph_name} Node Distribution' + (' (GCN Embedding)' if use_embedding else ' (Original Features)')
        filename = f'{graph_name}_{"gcn_embedding" if use_embedding else "original"}_distribution.png'
        
        print("\n开始可视化...")
        visualize_embeddings(
            reduced_features,
            normal_node_indices,
            target_node_indices,
            trigger_node_indices,
            title,
            save_path,
            filename
        )
        print(f"可视化结果已保存到: {os.path.join(save_path, filename)}")
    except Exception as e:
        print(f"降维失败: {str(e)}")
        print("详细错误信息:")
        import traceback
        traceback.print_exc()

def analyze_both_graphs(Gs, Gt, train_trigger_s, train_trigger_t, save_path):
    """分析两个图的节点分布"""
    # 分析原始特征分布
    print("\n分析原始特征分布...")
    analyze_node_distribution(Gs, train_trigger_s, train_trigger_s, 
                            os.path.join(save_path, 'feature_distribution'), 'Gs',
                            use_embedding=False)
    analyze_node_distribution(Gt, train_trigger_t, train_trigger_t, 
                            os.path.join(save_path, 'feature_distribution'), 'Gt',
                            use_embedding=False)
    
    # 使用GCN嵌入后分析
    print("\n分析GCN嵌入后的特征分布...")
    analyze_node_distribution(Gs, train_trigger_s, train_trigger_s, 
                            os.path.join(save_path, 'feature_distribution'), 'Gs',
                            use_embedding=True)
    analyze_node_distribution(Gt, train_trigger_t, train_trigger_t, 
                            os.path.join(save_path, 'feature_distribution'), 'Gt',
                            use_embedding=True) 