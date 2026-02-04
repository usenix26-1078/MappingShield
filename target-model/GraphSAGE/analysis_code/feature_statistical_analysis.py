import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cosine
import os

def numpy_to_python(obj):
    """
    将 NumPy 数据类型转换为 Python 原生类型
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    return obj

def calculate_statistics(normal_nodes_features, trigger_nodes_features, save_path, graph_name):
    """
    计算并可视化正常节点和触发器节点的统计特征
    
    Args:
        normal_nodes_features: 正常节点特征 [n_nodes, n_features]
        trigger_nodes_features: 触发器节点特征 [n_triggers, 3, n_features]
        save_path: 保存结果的路径
        graph_name: 图名称（'Gs' 或 'Gt'）
    """
    # 创建保存目录
    os.makedirs(os.path.join(save_path, graph_name), exist_ok=True)
    
    # 1. 计算正常节点的统计特征
    x0_mean = np.mean(normal_nodes_features, axis=0)
    x0_cov = np.cov(normal_nodes_features.T)
    
    # 1.1 从正常节点中随机采样两组节点计算基线KL散度
    n_normal = len(normal_nodes_features)
    if n_normal >= 1000:  # 确保有足够的节点进行采样
        indices = np.random.permutation(n_normal)
        normal_group1 = normal_nodes_features[indices[:500]]  # 第一组500个节点
        normal_group2 = normal_nodes_features[indices[500:1000]]  # 第二组500个节点
    else:
        # 如果节点数不足1000，则平均分成两组
        split_point = n_normal // 2
        indices = np.random.permutation(n_normal)
        normal_group1 = normal_nodes_features[indices[:split_point]]
        normal_group2 = normal_nodes_features[indices[split_point:]]
    
    # 1.2 计算正常节点组之间的KL散度
    n_features = normal_nodes_features.shape[1]
    baseline_kl_divergences = []
    baseline_kl_divergences_reverse = []
    
    for feature_idx in range(n_features):
        # 第一组正常节点的分布
        kde_normal1 = KernelDensity(kernel='gaussian')
        kde_normal1.fit(normal_group1[:, feature_idx:feature_idx+1])
        
        # 第二组正常节点的分布
        kde_normal2 = KernelDensity(kernel='gaussian')
        kde_normal2.fit(normal_group2[:, feature_idx:feature_idx+1])
        
        # 计算评估点
        x_eval = np.linspace(min(normal_group1[:, feature_idx].min(),
                               normal_group2[:, feature_idx].min()),
                           max(normal_group1[:, feature_idx].max(),
                               normal_group2[:, feature_idx].max()),
                           1000).reshape(-1, 1)
        
        # 计算两个分布
        p1 = np.exp(kde_normal1.score_samples(x_eval))
        p2 = np.exp(kde_normal2.score_samples(x_eval))
        
        # 计算两个方向的KL散度
        kl_baseline = entropy(p1 + 1e-10, p2 + 1e-10)
        kl_baseline_reverse = entropy(p2 + 1e-10, p1 + 1e-10)
        
        baseline_kl_divergences.append(float(kl_baseline))
        baseline_kl_divergences_reverse.append(float(kl_baseline_reverse))
    
    # 计算基线KL散度的均值和标准差
    mean_baseline_kl = np.mean(baseline_kl_divergences)
    std_baseline_kl = np.std(baseline_kl_divergences)
    mean_baseline_kl_reverse = np.mean(baseline_kl_divergences_reverse)
    std_baseline_kl_reverse = np.std(baseline_kl_divergences_reverse)
    
    # 1.1 计算正常节点与其平均值之间的相似度
    normal_cosine_sims = []
    normal_euclidean_dists = []
    for node_feat in normal_nodes_features:
        normal_cosine_sims.append(1 - cosine(node_feat, x0_mean))
        normal_euclidean_dists.append(np.linalg.norm(node_feat - x0_mean))
    
    normal_mean_cosine = np.mean(normal_cosine_sims)
    normal_mean_euclidean = np.mean(normal_euclidean_dists)
    normal_std_cosine = np.std(normal_cosine_sims)
    normal_std_euclidean = np.std(normal_euclidean_dists)
    
    # 2. 计算每个触发器与正常节点的相似度指标
    cosine_sims = []
    euclidean_dists = []
    
    for trigger_group in trigger_nodes_features:
        # 计算平均余弦相似度
        cos_sim = np.mean([1 - cosine(node_feat, x0_mean) for node_feat in trigger_group])
        cosine_sims.append(cos_sim)
        
        # 计算平均欧氏距离
        euc_dist = np.mean([np.linalg.norm(node_feat - x0_mean) for node_feat in trigger_group])
        euclidean_dists.append(euc_dist)
    
    # 3. 绘制相似度指标的分布
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(normal_cosine_sims, kde=True, color='blue', label=f'Normal Nodes (mean={normal_mean_cosine:.4f}, std={normal_std_cosine:.4f})')
    plt.axvline(x=normal_mean_cosine, color='blue', linestyle='--', label='Normal Mean')
    sns.histplot(cosine_sims, kde=True, color='red', label=f'Trigger Nodes (mean={np.mean(cosine_sims):.4f}, std={np.std(cosine_sims):.4f})')
    plt.title(f'{graph_name} - Cosine Similarity Distribution')
    plt.xlabel('Cosine Similarity')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    sns.histplot(normal_euclidean_dists, kde=True, color='blue', label=f'Normal Nodes (mean={normal_mean_euclidean:.4f}, std={normal_std_euclidean:.4f})')
    plt.axvline(x=normal_mean_euclidean, color='blue', linestyle='--', label='Normal Mean')
    sns.histplot(euclidean_dists, kde=True, color='red', label=f'Trigger Nodes (mean={np.mean(euclidean_dists):.4f}, std={np.std(euclidean_dists):.4f})')
    plt.title(f'{graph_name} - Euclidean Distance Distribution')
    plt.xlabel('Euclidean Distance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, graph_name, 'similarity_distributions.png'))
    plt.close()
    
    # 4. 计算KL散度
    n_features = normal_nodes_features.shape[1]
    kl_divergences = []
    kl_divergences_reverse = []  # 添加反向KL散度
    
    for feature_idx in range(n_features):
        # 正常节点特征分布
        kde_normal = KernelDensity(kernel='gaussian')
        kde_normal.fit(normal_nodes_features[:, feature_idx:feature_idx+1])
        
        # 触发器节点特征分布
        trigger_features = trigger_nodes_features.reshape(-1, n_features)[:, feature_idx]
        kde_trigger = KernelDensity(kernel='gaussian')
        kde_trigger.fit(trigger_features.reshape(-1, 1))
        
        # 计算KL散度
        x_eval = np.linspace(min(normal_nodes_features[:, feature_idx].min(),
                               trigger_features.min()),
                           max(normal_nodes_features[:, feature_idx].max(),
                               trigger_features.max()),
                           1000).reshape(-1, 1)
        
        p = np.exp(kde_normal.score_samples(x_eval))
        q = np.exp(kde_trigger.score_samples(x_eval))
        
        # KL(P||Q): 正常节点分布相对于触发器节点分布的KL散度
        kl_div = entropy(p + 1e-10, q + 1e-10)
        kl_divergences.append(float(kl_div))
        
        # KL(Q||P): 触发器节点分布相对于正常节点分布的KL散度
        kl_div_reverse = entropy(q + 1e-10, p + 1e-10)
        kl_divergences_reverse.append(float(kl_div_reverse))
    
    # 计算平均KL散度
    mean_kl = np.mean(kl_divergences)
    mean_kl_reverse = np.mean(kl_divergences_reverse)
    std_kl = np.std(kl_divergences)
    std_kl_reverse = np.std(kl_divergences_reverse)
    
    # 5. 可视化分析
    # KDE曲线
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    plt.figure(figsize=(20, 5 * n_rows))
    
    for feature_idx in range(n_features):
        plt.subplot(n_rows, n_cols, feature_idx + 1)
        
        # 正常节点分布
        sns.kdeplot(data=normal_nodes_features[:, feature_idx],
                   label='Normal Nodes', color='blue')
        
        # 触发器节点分布
        trigger_features = trigger_nodes_features.reshape(-1, n_features)[:, feature_idx]
        sns.kdeplot(data=trigger_features, label='Trigger Nodes', color='red')
        
        plt.title(f'Feature {feature_idx} Distribution')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, graph_name, 'feature_distributions.png'))
    plt.close()
    
    # Boxplot
    plt.figure(figsize=(20, 8))
    
    # 准备数据
    normal_data = []
    trigger_data = []
    labels = []
    
    for feature_idx in range(n_features):
        normal_data.extend(normal_nodes_features[:, feature_idx])
        trigger_data.extend(trigger_nodes_features.reshape(-1, n_features)[:, feature_idx])
        labels.extend([f'Normal-{feature_idx}'] * len(normal_nodes_features))
        labels.extend([f'Trigger-{feature_idx}'] * (len(trigger_nodes_features) * 3))
    
    data = normal_data + trigger_data
    
    sns.boxplot(x=labels, y=data)
    plt.xticks(rotation=45)
    plt.title(f'{graph_name} - Feature Distribution Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, graph_name, 'boxplot.png'))
    plt.close()
    
    # 7. 保存KL散度的可视化（更新后的版本）
    plt.figure(figsize=(20, 10))
    
    # KL(P||Q)
    plt.subplot(1, 2, 1)
    x = np.arange(n_features)
    width = 0.35
    
    plt.bar(x - width/2, kl_divergences, width, label='Trigger vs Normal', color='red', alpha=0.6)
    plt.bar(x + width/2, baseline_kl_divergences, width, label='Normal vs Normal', color='blue', alpha=0.6)
    
    plt.axhline(y=mean_kl, color='red', linestyle='--', 
                label=f'Trigger vs Normal Mean: {mean_kl:.4f}')
    plt.axhline(y=mean_baseline_kl, color='blue', linestyle='--', 
                label=f'Normal vs Normal Mean: {mean_baseline_kl:.4f}')
    
    plt.title(f'{graph_name} - KL Divergence Comparison (P to Q)')
    plt.xlabel('Feature Index')
    plt.ylabel('KL(P||Q)')
    plt.legend()
    
    # KL(Q||P)
    plt.subplot(1, 2, 2)
    plt.bar(x - width/2, kl_divergences_reverse, width, label='Trigger vs Normal', color='red', alpha=0.6)
    plt.bar(x + width/2, baseline_kl_divergences_reverse, width, label='Normal vs Normal', color='blue', alpha=0.6)
    
    plt.axhline(y=mean_kl_reverse, color='red', linestyle='--', 
                label=f'Trigger vs Normal Mean: {mean_kl_reverse:.4f}')
    plt.axhline(y=mean_baseline_kl_reverse, color='blue', linestyle='--', 
                label=f'Normal vs Normal Mean: {mean_baseline_kl_reverse:.4f}')
    
    plt.title(f'{graph_name} - KL Divergence Comparison (Q to P)')
    plt.xlabel('Feature Index')
    plt.ylabel('KL(Q||P)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, graph_name, 'kl_divergences_comparison.png'))
    plt.close()

    # 更新结果字典，添加基线KL散度
    results = {
        'normal_mean': x0_mean.tolist(),
        'normal_cov': x0_cov.tolist(),
        'normal_nodes_cosine_similarity_mean': float(normal_mean_cosine),
        'normal_nodes_cosine_similarity_std': float(normal_std_cosine),
        'normal_nodes_euclidean_distance_mean': float(normal_mean_euclidean),
        'normal_nodes_euclidean_distance_std': float(normal_std_euclidean),
        'trigger_cosine_similarity_mean': float(np.mean(cosine_sims)),
        'trigger_cosine_similarity_std': float(np.std(cosine_sims)),
        'trigger_euclidean_distance_mean': float(np.mean(euclidean_dists)),
        'trigger_euclidean_distance_std': float(np.std(euclidean_dists)),
        'kl_divergence': {
            'per_feature': {
                'normal_to_trigger': kl_divergences,  # KL(P||Q)
                'trigger_to_normal': kl_divergences_reverse,  # KL(Q||P)
                'normal_to_normal': baseline_kl_divergences,  # 基线 KL(P||Q)
                'normal_to_normal_reverse': baseline_kl_divergences_reverse,  # 基线 KL(Q||P)
            },
            'average': {
                'normal_to_trigger_mean': float(mean_kl),
                'normal_to_trigger_std': float(std_kl),
                'trigger_to_normal_mean': float(mean_kl_reverse),
                'trigger_to_normal_std': float(std_kl_reverse),
                'normal_to_normal_mean': float(mean_baseline_kl),
                'normal_to_normal_std': float(std_baseline_kl),
                'normal_to_normal_reverse_mean': float(mean_baseline_kl_reverse),
                'normal_to_normal_reverse_std': float(std_baseline_kl_reverse),
            }
        }
    }
    
    # 确保所有数值都转换为 Python 原生类型
    results = numpy_to_python(results)
    
    import json
    with open(os.path.join(save_path, graph_name, 'statistics.json'), 'w') as f:
        json.dump(results, f, indent=4)

def analyze_graphs(Gs, Gt, train_trigger_s, train_trigger_t, save_path):
    """
    分析两个图的正常节点和触发器节点特征
    
    Args:
        Gs: 源图
        Gt: 目标图
        train_trigger_s: 源图触发器节点字典
        train_trigger_t: 目标图触发器节点字典
        save_path: 保存结果的路径
    """
    # 为两个图分别进行分析
    for G, triggers, name in [(Gs, train_trigger_s, 'Gs'), (Gt, train_trigger_t, 'Gt')]:
        # 获取所有节点索引
        all_nodes = set(range(G.x.shape[0]))
        
        # 获取触发器节点和新增节点的索引
        trigger_nodes = set()
        for node, indices in triggers.items():
            trigger_nodes.add(node)  # 原始节点
            trigger_nodes.update(indices.tolist())  # 触发器节点
        
        # 获取正常节点（排除触发器节点和新增节点）
        normal_nodes = list(all_nodes - trigger_nodes)
        
        # 随机采样1000个正常节点
        if len(normal_nodes) > 1000:
            normal_nodes = np.random.choice(normal_nodes, 1000, replace=False)
        
        # 提取特征
        normal_features = G.x[normal_nodes].cpu().numpy()
        
        # 提取触发器特征
        trigger_features = []
        for _, indices in triggers.items():
            trigger_group = G.x[indices].cpu().numpy()
            trigger_features.append(trigger_group)
        
        trigger_features = np.array(trigger_features)
        
        # 进行统计分析
        calculate_statistics(normal_features, trigger_features, save_path, name) 