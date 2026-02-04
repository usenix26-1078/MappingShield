import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils import to_networkx
from community import community_louvain
import os
from collections import defaultdict
from tqdm import tqdm
import json

def analyze_triangles(G, trigger_nodes, save_path, graph_name):
    """
    分析图中的三角形（K3）分布
    
    Args:
        G: networkx图对象
        trigger_nodes: 触发器节点集合
        save_path: 结果保存路径
        graph_name: 图名称（'Gs' 或 'Gt'）
    """
    # 计算全图三角形
    all_triangles = sum(nx.triangles(G).values()) // 3
    
    # 计算每个触发器子图内部的三角形
    trigger_triangles = 0
    trigger_groups = defaultdict(set)
    
    # 首先按照目标节点分组触发器节点
    for node in trigger_nodes:
        # 找到与该节点相连的其他触发器节点
        neighbors = set(G.neighbors(node)) & trigger_nodes
        if len(neighbors) > 0:  # 如果有相连的触发器节点
            # 遍历所有已知的触发器组
            assigned = False
            for group in trigger_groups.values():
                if node in group or any(n in group for n in neighbors):
                    group.add(node)
                    group.update(neighbors)
                    assigned = True
                    break
            if not assigned:
                # 创建新的触发器组
                group_id = len(trigger_groups)
                trigger_groups[group_id] = {node} | neighbors

    # 计算每个触发器组内的三角形数量
    for group in trigger_groups.values():
        if len(group) >= 3:  # 只有当组内节点数大于等于3时才可能形成三角形
            subgraph = G.subgraph(group)
            group_triangles = sum(nx.triangles(subgraph).values()) // 3
            trigger_triangles += group_triangles
    
    # 计算比例
    ratio = trigger_triangles / all_triangles if all_triangles > 0 else 0
    
    # 保存结果
    results = {
        'total_triangles': all_triangles,
        'trigger_triangles': trigger_triangles,
        'trigger_groups_count': len(trigger_groups),
        'trigger_groups_sizes': [len(group) for group in trigger_groups.values()],
        'ratio': float(ratio)
    }
    
    os.makedirs(os.path.join(save_path, graph_name), exist_ok=True)
    with open(os.path.join(save_path, graph_name, 'triangle_analysis.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"{graph_name} Triangle Analysis:")
    print(f"Total triangles: {all_triangles}")
    print(f"Trigger triangles: {trigger_triangles}")
    print(f"Number of trigger groups: {len(trigger_groups)}")
    print(f"Trigger group sizes: {[len(group) for group in trigger_groups.values()]}")
    print(f"Ratio: {ratio:.4f}")

def calculate_node_metrics(G, node, betweenness_dict=None, pagerank_dict=None):
    """计算单个节点的结构指标"""
    metrics = {
        'degree': G.degree(node),
        'clustering': nx.clustering(G, node)
    }
    
    # 平均邻居度
    neighbors = list(G.neighbors(node))
    metrics['avg_neighbor_degree'] = np.mean([G.degree(n) for n in neighbors]) if neighbors else 0
    
    # 使用预计算的中心性指标
    if betweenness_dict is not None:
        metrics['betweenness'] = betweenness_dict.get(node, 0.0)
    if pagerank_dict is not None:
        metrics['pagerank'] = pagerank_dict.get(node, 0.0)
    
    return metrics

def analyze_structural_metrics(G, trigger_nodes, save_path, graph_name):
    """
    分析节点结构指标
    
    Args:
        G: networkx图对象
        trigger_nodes: 触发器节点集合
        save_path: 结果保存路径
        graph_name: 图名称
    """
    print(f"\n分析 {graph_name} 的节点结构指标...")
    
    # 获取普通节点
    normal_nodes = set(G.nodes()) - set(trigger_nodes)
    
    # 随机采样1000个普通节点
    sampled_normal_nodes = np.random.choice(list(normal_nodes), 
                                          min(1000, len(normal_nodes)), 
                                          replace=False)
    
    # 预计算耗时的中心性指标
    print("预计算PageRank（这可能需要几分钟）...")
    pagerank_dict = nx.pagerank(G)
    
    print("预计算介数中心性（使用采样方法，这可能需要几分钟）...")
    if len(G) > 1000:
        betweenness_dict = nx.betweenness_centrality(G, k=100)
    else:
        betweenness_dict = nx.betweenness_centrality(G)
    
    # 计算节点指标
    normal_metrics = defaultdict(list)
    trigger_metrics = defaultdict(list)
    
    print("计算普通节点指标...")
    for node in tqdm(sampled_normal_nodes):
        metrics = calculate_node_metrics(G, node, betweenness_dict, pagerank_dict)
        for k, v in metrics.items():
            normal_metrics[k].append(v)
    
    print("计算触发器节点指标...")
    for node in tqdm(trigger_nodes):
        metrics = calculate_node_metrics(G, node, betweenness_dict, pagerank_dict)
        for k, v in metrics.items():
            trigger_metrics[k].append(v)
    
    # 计算统计量
    stats = {}
    for metric in normal_metrics.keys():
        stats[metric] = {
            'normal': {
                'mean': float(np.mean(normal_metrics[metric])),
                'std': float(np.std(normal_metrics[metric])),
                'median': float(np.median(normal_metrics[metric]))
            },
            'trigger': {
                'mean': float(np.mean(trigger_metrics[metric])),
                'std': float(np.std(trigger_metrics[metric])),
                'median': float(np.median(trigger_metrics[metric]))
            }
        }
    
    # 保存统计结果
    os.makedirs(os.path.join(save_path, graph_name), exist_ok=True)
    with open(os.path.join(save_path, graph_name, 'structural_metrics.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    # 绘制分布对比图
    metrics = list(normal_metrics.keys())
    n_metrics = len(metrics)
    n_rows = (n_metrics + 2) // 3  # 使用3列布局
    n_cols = min(3, n_metrics)
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(n_rows, n_cols, i)
        
        # 检查是否有方差
        normal_std = np.std(normal_metrics[metric])
        trigger_std = np.std(trigger_metrics[metric])
        
        if normal_std > 0:
            sns.histplot(data=normal_metrics[metric], label='Normal', color='blue', alpha=0.6, stat='density', bins=30)
        else:
            plt.axvline(np.mean(normal_metrics[metric]), color='blue', label='Normal', alpha=0.6)
            
        if trigger_std > 0:
            sns.histplot(data=trigger_metrics[metric], label='Trigger', color='red', alpha=0.6, stat='density', bins=30)
        else:
            plt.axvline(np.mean(trigger_metrics[metric]), color='red', label='Trigger', alpha=0.6)
        
        plt.title(f'{metric.replace("_", " ").title()}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        
        # 添加均值线
        plt.axvline(np.mean(normal_metrics[metric]), color='blue', linestyle='--', alpha=0.3)
        plt.axvline(np.mean(trigger_metrics[metric]), color='red', linestyle='--', alpha=0.3)
        
        if i == 1:  # 只在第一个子图显示图例
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, graph_name, 'metric_distributions.png'))
    plt.close()

def visualize_ego_networks(G, trigger_dict, trigger_nodes, save_path, graph_name):
    """
    可视化带有触发器的目标节点的2-hop ego network
    
    Args:
        G: networkx图对象
        trigger_dict: 触发器节点字典 {target_node: trigger_nodes}
        trigger_nodes: 所有触发器节点集合
        save_path: 结果保存路径
        graph_name: 图名称
    """
    # 获取所有目标节点（带有触发器的节点）
    target_nodes = list(trigger_dict.keys())
    print(f"\n{graph_name} 目标节点统计:")
    print(f"总目标节点数: {len(target_nodes)}")
    
    # 如果目标节点太多，随机采样一部分
    max_nodes = 49  # 7x7网格最多显示49个子图
    if len(target_nodes) > max_nodes:
        sampled_nodes = np.random.choice(target_nodes, max_nodes, replace=False)
    else:
        sampled_nodes = target_nodes
    
    # 创建一个大图用于可视化
    plt.figure(figsize=(20, 20))
    
    # 计算子图布局：7x7网格（最多显示49个子图）
    n_rows = 7
    n_cols = 7
    
    for idx, node in enumerate(sampled_nodes):
        if idx >= n_rows * n_cols:
            break
            
        # 获取2-hop邻居
        ego_nodes = {node}  # 中心节点
        first_hop = set(G.neighbors(node))  # 1-hop邻居
        ego_nodes.update(first_hop)
        
        # 获取2-hop邻居
        second_hop = set()
        for neighbor in first_hop:
            second_hop.update(G.neighbors(neighbor))
        ego_nodes.update(second_hop)
        
        # 打印节点统计信息
        print(f"\n目标节点 {node} 的统计信息:")
        print(f"1-hop邻居数量: {len(first_hop)}")
        print(f"2-hop邻居数量: {len(second_hop)}")
        print(f"总邻居数量: {len(ego_nodes) - 1}")  # 减去中心节点
        
        # 统计触发器节点数量
        trigger_count = sum(1 for n in ego_nodes if n in trigger_nodes)
        print(f"ego network中的触发器节点数量: {trigger_count}")
        
        # 提取子图
        subgraph = G.subgraph(ego_nodes)
        
        # 设置节点颜色：红色为触发器节点，黄色为中心节点，蓝色为普通节点
        colors = []
        for n in subgraph.nodes():
            if n == node:
                colors.append('yellow')  # 中心节点（目标节点）
            elif n in trigger_nodes:
                colors.append('red')     # 触发器节点
            else:
                colors.append('blue')    # 普通节点
        
        # 添加子图
        plt.subplot(n_rows, n_cols, idx + 1)
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, 
               node_color=colors,
               node_size=20,  # 减小节点大小
               with_labels=False,  # 不显示标签
               width=0.5)  # 减小边的宽度
        plt.title(f'Target Node {node}\nNeighbors: {len(ego_nodes)-1}', fontsize=8)
    
    plt.suptitle(f'{graph_name} Target Nodes Ego Networks', fontsize=16)
    plt.tight_layout()
    os.makedirs(os.path.join(save_path, graph_name), exist_ok=True)
    plt.savefig(os.path.join(save_path, graph_name, 'ego_networks.png'), dpi=300)
    plt.close()

def analyze_communities(G, trigger_dict, trigger_nodes, save_path, graph_name):
    """社区分析
    
    Args:
        G: networkx图对象
        trigger_dict: 触发器节点字典
        trigger_nodes: 所有触发器节点集合
        save_path: 结果保存路径
        graph_name: 图名称
    """
    # 使用Louvain方法进行社区检测
    communities = community_louvain.best_partition(G)
    
    # 统计每个社区的节点数量和触发器节点比例
    community_stats = {}
    for node, comm_id in communities.items():
        if comm_id not in community_stats:
            community_stats[comm_id] = {'total': 0, 'trigger': 0}
        community_stats[comm_id]['total'] += 1
        
        # 检查是否为触发器节点
        if node in trigger_nodes:
            community_stats[comm_id]['trigger'] += 1
    
    # 计算统计信息
    communities_with_triggers = 0
    high_trigger_ratio_communities = 0
    for stats in community_stats.values():
        if stats['trigger'] > 0:
            communities_with_triggers += 1
            if stats['trigger'] / stats['total'] > 0.7:
                high_trigger_ratio_communities += 1
    
    total_communities = len(community_stats)
    
    # 可视化社区结构
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # 绘制所有节点，颜色按社区区分
    nx.draw_networkx_nodes(G, pos,
                          node_color=list(communities.values()),
                          node_size=20,
                          cmap=plt.cm.tab20)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    # 标记触发器节点：用红色边框而不是覆盖颜色
    trigger_pos = {node: pos[node] for node in trigger_nodes}
    nx.draw_networkx_nodes(G,
                          pos,
                          nodelist=list(trigger_nodes),
                          node_color='none',  # 透明填充
                          edgecolors='red',   # 红色边框
                          linewidths=2,       # 边框宽度
                          node_size=30,       # 稍微大一点以便看清边框
                          label='Trigger Nodes')
    
    # 添加统计信息到标题
    plt.title(f'{graph_name} Community Structure\n'
             f'Total Communities: {total_communities}\n'
             f'Communities with Triggers: {communities_with_triggers} ({communities_with_triggers/total_communities*100:.1f}%)\n'
             f'Communities with >70% Triggers: {high_trigger_ratio_communities} ({high_trigger_ratio_communities/total_communities*100:.1f}%)')
    
    plt.legend()
    
    # 保存结果
    os.makedirs(os.path.join(save_path, 'community_analysis'), exist_ok=True)
    plt.savefig(os.path.join(save_path, 'community_analysis', f'{graph_name}_community_structure.png'))
    plt.close()
    
    # 保存统计信息到JSON文件
    stats = {
        'total_communities': total_communities,
        'communities_with_triggers': communities_with_triggers,
        'communities_with_triggers_ratio': communities_with_triggers/total_communities,
        'high_trigger_ratio_communities': high_trigger_ratio_communities,
        'high_trigger_ratio_communities_ratio': high_trigger_ratio_communities/total_communities
    }
    
    with open(os.path.join(save_path, 'community_analysis', f'{graph_name}_community_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)

def analyze_graph_structure(pyg_graph, trigger_dict, save_path, graph_name):
    """
    主函数：执行所有结构分析
    
    Args:
        pyg_graph: PyTorch Geometric图对象
        trigger_dict: 触发器节点字典
        save_path: 结果保存路径
        graph_name: 图名称
    """
    print(f"\n分析 {graph_name} 的图结构...")
    
    # 转换为networkx图
    G = to_networkx(pyg_graph, to_undirected=True)
    
    # 获取所有触发器节点（只包括触发器节点，不包括目标节点）
    trigger_nodes = set()
    for target_node, trigger_indices in trigger_dict.items():
        trigger_nodes.update(trigger_indices.tolist())
    
    # 1. 分析三角形
    print("\n分析三角形分布...")
    analyze_triangles(G, trigger_nodes, save_path, graph_name)
    
    # 2. 分析结构指标
    print("\n分析节点结构指标...")
    analyze_structural_metrics(G, trigger_nodes, save_path, graph_name)
    
    # 3. 可视化ego networks
    print("\n生成ego networks可视化...")
    visualize_ego_networks(G, trigger_dict, trigger_nodes, save_path, graph_name)
    
    # 4. 社区分析
    print("\n进行社区分析...")
    analyze_communities(G, trigger_dict, trigger_nodes, save_path, graph_name)

def analyze_both_graphs(Gs, Gt, train_trigger_s, train_trigger_t, save_path):
    """
    分析两个图的结构
    
    Args:
        Gs: 源图
        Gt: 目标图
        train_trigger_s: 源图触发器节点字典
        train_trigger_t: 目标图触发器节点字典
        save_path: 结果保存路径
    """
    # 分析源图
    analyze_graph_structure(Gs, train_trigger_s, save_path, 'Gs')
    
    # 分析目标图
    analyze_graph_structure(Gt, train_trigger_t, save_path, 'Gt') 