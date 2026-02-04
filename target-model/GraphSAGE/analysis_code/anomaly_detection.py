import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.metrics import roc_auc_score
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class DOMINANT(nn.Module):
    """
    DOMINANT模型实现
    基于GCN的图自编码器进行异常检测
    """
    def __init__(self, num_features, hidden_dim=64, num_layers=2):
        super(DOMINANT, self).__init__()
        self.num_layers = num_layers
        
        # 编码器
        self.encoder = nn.ModuleList()
        self.encoder.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.encoder.append(GCNConv(hidden_dim, hidden_dim))
            
        # 结构重建解码器
        self.struct_decoder = nn.Linear(hidden_dim, hidden_dim)
        
        # 属性重建解码器
        self.attri_decoder = nn.Linear(hidden_dim, num_features)
        
    def encode(self, x, edge_index):
        h = x
        for layer in self.encoder:
            h = layer(h, edge_index)
            h = F.relu(h)
        return h
        
    def decode(self, z):
        # 结构重建
        struct_reconstructed = torch.sigmoid(self.struct_decoder(z) @ z.t())
        # 属性重建
        attri_reconstructed = self.attri_decoder(z)
        return struct_reconstructed, attri_reconstructed
        
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        struct_reconstructed, attri_reconstructed = self.decode(z)
        return struct_reconstructed, attri_reconstructed

class AnomalyDAE(nn.Module):
    """
    AnomalyDAE模型实现
    基于深度自编码器的图异常检测
    """
    def __init__(self, num_features, hidden_dim=64):
        super(AnomalyDAE, self).__init__()
        
        # 编码器
        self.enc1 = GCNConv(num_features, hidden_dim * 2)
        self.enc2 = GCNConv(hidden_dim * 2, hidden_dim)
        
        # 解码器
        self.dec1 = GCNConv(hidden_dim, hidden_dim * 2)
        self.dec2 = GCNConv(hidden_dim * 2, num_features)
        
    def forward(self, x, edge_index):
        # 编码
        h = F.relu(self.enc1(x, edge_index))
        z = F.relu(self.enc2(h, edge_index))
        
        # 解码
        h = F.relu(self.dec1(z, edge_index))
        x_hat = self.dec2(h, edge_index)
        return x_hat

def train_dominant(model, data, device, epochs=100):
    """训练DOMINANT模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in tqdm(range(epochs), desc="Training DOMINANT"):
        optimizer.zero_grad()
        struct_reconstructed, attri_reconstructed = model(data.x.to(device), data.edge_index.to(device))
        
        # 计算结构重建损失
        adj = torch.zeros((data.num_nodes, data.num_nodes), device=device)
        adj[data.edge_index[0], data.edge_index[1]] = 1
        struct_loss = F.binary_cross_entropy_with_logits(struct_reconstructed, adj)
        
        # 计算属性重建损失
        attri_loss = F.mse_loss(attri_reconstructed, data.x.to(device))
        
        # 总损失
        loss = struct_loss + attri_loss
        loss.backward()
        optimizer.step()

def train_anomalydae(model, data, device, epochs=100):
    """训练AnomalyDAE模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in tqdm(range(epochs), desc="Training AnomalyDAE"):
        optimizer.zero_grad()
        x_hat = model(data.x.to(device), data.edge_index.to(device))
        loss = F.mse_loss(x_hat, data.x.to(device))
        loss.backward()
        optimizer.step()

def get_anomaly_scores_dominant(model, data, device):
    """计算DOMINANT的异常分数"""
    model.eval()
    with torch.no_grad():
        struct_reconstructed, attri_reconstructed = model(data.x.to(device), data.edge_index.to(device))
        
        # 计算结构重建误差
        adj = torch.zeros((data.num_nodes, data.num_nodes), device=device)
        adj[data.edge_index[0], data.edge_index[1]] = 1
        struct_errors = F.binary_cross_entropy_with_logits(struct_reconstructed, adj, reduction='none')
        struct_scores = struct_errors.mean(dim=1)
        
        # 计算属性重建误差
        attri_errors = F.mse_loss(attri_reconstructed, data.x.to(device), reduction='none')
        attri_scores = attri_errors.mean(dim=1)
        
        # 综合分数
        anomaly_scores = struct_scores + attri_scores
        return anomaly_scores.cpu().numpy()

def get_anomaly_scores_anomalydae(model, data, device):
    """计算AnomalyDAE的异常分数"""
    model.eval()
    with torch.no_grad():
        x_hat = model(data.x.to(device), data.edge_index.to(device))
        reconstruction_errors = F.mse_loss(x_hat, data.x.to(device), reduction='none')
        anomaly_scores = reconstruction_errors.mean(dim=1)
        return anomaly_scores.cpu().numpy()

def evaluate_anomaly_detection(anomaly_scores, trigger_nodes, save_path, model_name):
    """评估异常检测结果"""
    # 创建标签（触发器节点为1，普通节点为0）
    labels = np.zeros(len(anomaly_scores))
    labels[list(trigger_nodes)] = 1
    
    # 计算AUC
    auc = roc_auc_score(labels, anomaly_scores)
    
    # 获取排序后的索引
    sorted_indices = np.argsort(anomaly_scores)[::-1]
    
    # 计算Precision@K
    k_values = [5, 10]
    precision_at_k = {}
    for k in k_values:
        top_k_indices = sorted_indices[:k]
        precision_at_k[k] = np.sum(labels[top_k_indices]) / k
    
    # 获取触发器节点的排名
    trigger_rankings = []
    for trigger_node in trigger_nodes:
        ranking = np.where(sorted_indices == trigger_node)[0][0] + 1
        trigger_rankings.append(ranking)
    
    # 计算CDF
    total_nodes = len(anomaly_scores)
    trigger_ranks = np.array(trigger_rankings)
    normal_ranks = np.array([i for i in range(1, total_nodes + 1) if i not in trigger_ranks])
    
    # 计算触发器节点的CDF
    trigger_cdf = np.zeros(total_nodes + 1)
    for rank in trigger_ranks:
        trigger_cdf[rank:] += 1
    trigger_cdf = trigger_cdf / len(trigger_ranks)
    
    # 计算普通节点的CDF
    normal_cdf = np.zeros(total_nodes + 1)
    for rank in normal_ranks:
        normal_cdf[rank:] += 1
    normal_cdf = normal_cdf / len(normal_ranks)
    
    # 绘制CDF图
    plt.figure(figsize=(10, 6))
    x = np.arange(total_nodes + 1)
    plt.plot(x, trigger_cdf, 'r-', label=f'Trigger Nodes (AUC={auc:.4f}, P@5={precision_at_k[5]:.4f}, P@10={precision_at_k[10]:.4f})')
    plt.plot(x, normal_cdf, 'b-', label='Normal Nodes')
    
    plt.xlabel('Rank')
    plt.ylabel('Cumulative Distribution')
    plt.title(f'{model_name} - Rank Distribution CDF')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图片
    plt.savefig(os.path.join(save_path, f'{model_name}_rank_cdf.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存结果
    results = {
        'model_name': model_name,
        'auc': float(auc),
        'precision@5': float(precision_at_k[5]),
        'precision@10': float(precision_at_k[10]),
        'trigger_rankings': {
            'mean': float(np.mean(trigger_rankings)),
            'median': float(np.median(trigger_rankings)),
            'min': int(np.min(trigger_rankings)),
            'max': int(np.max(trigger_rankings)),
            'individual_rankings': [int(r) for r in trigger_rankings]
        },
        'trigger_scores': {
            str(node): float(anomaly_scores[node]) for node in trigger_nodes
        }
    }
    
    # 打印结果
    print(f"\n{model_name} Results:")
    print(f"AUC: {auc:.4f}")
    print(f"Precision@5: {precision_at_k[5]:.4f}")
    print(f"Precision@10: {precision_at_k[10]:.4f}")
    print(f"Trigger Rankings - Mean: {np.mean(trigger_rankings):.1f}, "
          f"Median: {np.median(trigger_rankings)}, "
          f"Range: [{np.min(trigger_rankings)}, {np.max(trigger_rankings)}]")
    
    return results

def analyze_anomalies(data, trigger_nodes, save_path, graph_name):
    """对图进行异常检测分析"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 确保保存路径存在
    os.makedirs(os.path.join(save_path, graph_name), exist_ok=True)
    
    results = {}
    
    # DOMINANT模型
    print(f"\nRunning DOMINANT on {graph_name}...")
    dominant = DOMINANT(num_features=data.x.size(1))
    train_dominant(dominant, data, device)
    dominant_scores = get_anomaly_scores_dominant(dominant, data, device)
    results['DOMINANT'] = evaluate_anomaly_detection(
        dominant_scores, trigger_nodes,
        os.path.join(save_path, graph_name),
        'DOMINANT'
    )
    
    # AnomalyDAE模型
    print(f"\nRunning AnomalyDAE on {graph_name}...")
    anomalydae = AnomalyDAE(num_features=data.x.size(1))
    train_anomalydae(anomalydae, data, device)
    anomalydae_scores = get_anomaly_scores_anomalydae(anomalydae, data, device)
    results['AnomalyDAE'] = evaluate_anomaly_detection(
        anomalydae_scores, trigger_nodes,
        os.path.join(save_path, graph_name),
        'AnomalyDAE'
    )
    
    # 保存结果
    import json
    with open(os.path.join(save_path, graph_name, 'anomaly_detection_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def analyze_both_graphs(Gs, Gt, train_trigger_s, train_trigger_t, save_path):
    """分析两个图的异常检测结果"""
    # 获取触发器节点集合
    trigger_nodes_s = set()
    for target_node, trigger_indices in train_trigger_s.items():
        trigger_nodes_s.update(trigger_indices.tolist())
    
    trigger_nodes_t = set()
    for target_node, trigger_indices in train_trigger_t.items():
        trigger_nodes_t.update(trigger_indices.tolist())
    
    # 分析源图
    print("\nAnalyzing anomalies in source graph (Gs)...")
    results_s = analyze_anomalies(Gs, trigger_nodes_s, save_path, 'Gs')
    
    # 分析目标图
    print("\nAnalyzing anomalies in target graph (Gt)...")
    results_t = analyze_anomalies(Gt, trigger_nodes_t, save_path, 'Gt')
    
    return results_s, results_t 