import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import torch

def convert_to_serializable(obj):
    """将NumPy和PyTorch类型转换为可JSON序列化的Python原生类型"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def similarity_analysis(G, trigger, flag, output_folder):
    """分析触发器子图与原始节点的相似性
    Args:
        G: 图数据
        trigger: 字典，键为原始节点索引，值为对应的触发器节点索引列表（3个节点）
        flag: 标识符（'s'或't'）
        output_folder: 输出文件夹路径
    """
    # 存储每个子图的统计值
    subgraph_node_trigger_means = []  # 每个子图的节点-触发器相似度均值
    subgraph_internal_means = []      # 每个子图的内部相似度均值
    
    for orig_node, trigger_nodes in trigger.items():
        # 获取原始节点特征
        orig_feature = G.x[orig_node].detach().cpu().numpy().reshape(1, -1)
        
        # 获取三个触发器节点的特征
        trigger_features = []
        for t_node in trigger_nodes:
            t_feature = G.x[t_node].detach().cpu().numpy().reshape(1, -1)
            trigger_features.append(t_feature)
            
        # 1. 计算原始节点与其对应三个触发器节点的相似度均值
        node_sims = []
        for t_feature in trigger_features:
            sim = float(cosine_similarity(orig_feature, t_feature)[0][0])
            node_sims.append(sim)
        subgraph_node_trigger_means.append(float(np.mean(node_sims)))
        
        # 2. 计算触发器子图内部三个节点之间的相似度均值
        internal_sims = []
        for i in range(len(trigger_features)):
            for j in range(i+1, len(trigger_features)):
                sim = float(cosine_similarity(trigger_features[i], trigger_features[j])[0][0])
                internal_sims.append(sim)
        subgraph_internal_means.append(float(np.mean(internal_sims)))

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 保存统计结果
    with open(os.path.join(output_folder, f'similarity_analysis_{flag}.txt'), 'w') as file:
        # 1. 节点-触发器相似度统计（子图级别）
        file.write(f"图{flag}的节点-触发器相似度统计（子图级别）：\n")
        file.write(f"子图均值的均值: {float(np.mean(subgraph_node_trigger_means)):.4f}\n")
        file.write(f"子图均值的最大值: {float(np.max(subgraph_node_trigger_means)):.4f}\n")
        file.write(f"子图均值的最小值: {float(np.min(subgraph_node_trigger_means)):.4f}\n")
        file.write(f"子图均值的标准差: {float(np.std(subgraph_node_trigger_means)):.4f}\n\n")
        
        # 2. 触发器内部相似度统计（子图级别）
        file.write(f"图{flag}的触发器内部相似度统计（子图级别）：\n")
        file.write(f"子图均值的均值: {float(np.mean(subgraph_internal_means)):.4f}\n")
        file.write(f"子图均值的最大值: {float(np.max(subgraph_internal_means)):.4f}\n")
        file.write(f"子图均值的最小值: {float(np.min(subgraph_internal_means)):.4f}\n")
        file.write(f"子图均值的标准差: {float(np.std(subgraph_internal_means)):.4f}\n")

    # 可视化（展示子图级别均值的分布）
    plt.figure(figsize=(12, 5))
    
    # 1. 节点-触发器相似度均值分布
    plt.subplot(1, 2, 1)
    plt.hist(subgraph_node_trigger_means, bins=20, edgecolor='black', density=True)
    plt.title(f'G{flag} Node-Trigger Mean Similarity\nDistribution (Subgraph Level)')
    plt.xlabel('Mean Similarity')
    plt.ylabel('Frequency')
    
    # 2. 触发器内部相似度均值分布
    plt.subplot(1, 2, 2)
    plt.hist(subgraph_internal_means, bins=20, edgecolor='black', density=True)
    plt.title(f'G{flag} Trigger Internal Mean Similarity\nDistribution (Subgraph Level)')
    plt.xlabel('Mean Similarity')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'similarity_distribution_{flag}.png'))
    plt.close()