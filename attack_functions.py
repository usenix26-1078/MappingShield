import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import degree
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from torch_geometric.utils import to_networkx
from data_loader import load_graph_from_npz
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import random

def torch_geometric_pagerank(data):
    """
    使用 Power Iteration 方法计算 Pagerank。
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    alpha = 0.85  # Pagerank 的阻尼因子
    tol = 1e-6  # 收敛阈值

    # 初始化 Pagerank 分数
    pr = torch.full((num_nodes,), 1.0 / num_nodes, dtype=torch.float)
    out_degree = degree(edge_index[0], num_nodes=num_nodes)

    # 避免分母为零
    out_degree[out_degree == 0] = 1.0

    for _ in range(100):  # 最多迭代 100 次
        prev_pr = pr.clone()
        pr = torch.zeros_like(pr)

        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i]
            pr[dst] += prev_pr[src] / out_degree[src]

        pr = alpha * pr + (1 - alpha) / num_nodes

        # 检查收敛
        if torch.norm(pr - prev_pr, p=1) < tol:
            break

    return pr.cpu().numpy()

def calculate_clustering_coefficient(data):
    """
    使用 networkx 计算聚类系数
    :param data: torch_geometric.data.Data 对象
    :return: 聚类系数数组
    """
    # 转换为 networkx 图
    G = to_networkx(data, to_undirected=True)
    
    # 计算每个节点的聚类系数
    clustering_coeff = nx.clustering(G)
    
    # 转换为 numpy 数组
    return np.array([clustering_coeff[node] for node in range(data.num_nodes)])


def calculate_node_features(data):
    """
    计算图的节点特征，包括：
    - 度数
    - PageRank
    - 邻居平均度数
    - 聚类系数
    - K-core 指数
    - 接近中心性
    - 介数中心性
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    
    # 度数
    degree_centrality = degree(edge_index[0], num_nodes=num_nodes).cpu().numpy()
    
    # PageRank (示例伪实现，需要替换为 torch_geometric 的实现)
    pagerank = torch_geometric_pagerank(data)
    
    # 邻居平均度数
    neighbor_degree = np.zeros(num_nodes)
    for i in range(num_nodes):
        # 找到节点 i 的邻居
        neighbors = edge_index[1][edge_index[0] == i]
        
        # 计算邻居的平均度数
        neighbor_degree[i] = (
            np.mean(degree(edge_index[0], num_nodes=num_nodes)[neighbors].cpu().numpy())
            if len(neighbors) > 0
            else 0
        )

    
    # 聚类系数 (改为调用 networkx 或手动实现的版本)
    cluster_coeff = calculate_clustering_coefficient(data)
    
    # K-core 指数 (伪实现，需自定义或使用现成库)
    k_core_index = np.random.uniform(0, 10, num_nodes)  # 需替换为实际的 K-core 算法实现
    
    # 接近中心性和介数中心性 (需要 graph-tool 或 networkx)
    closeness_centrality = np.random.uniform(0, 1, num_nodes)  # 伪实现
    betweenness_centrality = np.random.uniform(0, 1, num_nodes)  # 伪实现

    return {
        'degree': degree_centrality,
        'pagerank': pagerank,
        'neighbor_degree': neighbor_degree,
        'cluster_coeff': cluster_coeff,
        'k_core': k_core_index,
        'closeness': closeness_centrality,
        'betweenness': betweenness_centrality
    }


def select_least_similar_node_pairs(data1, data2, train_data, top_n, weights):
    """
    选择攻击节点对，仅从 train_data 里 label=0 的节点对中选最不像的 top_n 个。
    
    :param data1: 图1数据
    :param data2: 图2数据
    :param train_data: 训练集，第一行是 Gs 的节点，第二行是 Gt 的节点，第三行是 label（0 或 1）
    :param top_n: 选择 top_n 个最不像的节点对
    :param weights: 各指标的权重，字典形式，例如：
        {
            'degree': 0.2,
            'pagerank': 0.2,
            'neighbor_degree': 0.2,
            'cluster_coeff': -0.2,
            'k_core': 0.1,
            'closeness': 0.1,
            'betweenness': 0.1
        }
    """

    # 1. 计算图1和图2的节点特征
    features_data1 = calculate_node_features(data1)
    features_data2 = calculate_node_features(data2)
    print('Finish node metric calculation')

    # 2. 取出 train_data 里 label = 0 的节点对
    mask = (train_data[2] == 0)  # 找到 label=0 的索引
    candidate_nodes_data1 = train_data[0][mask]  # Gs 中的节点
    candidate_nodes_data2 = train_data[1][mask]  # Gt 中的节点

    print(f"Found {len(candidate_nodes_data1)} candidate node pairs.")

    # 3. 计算这些节点的特征
    feature_matrix_data1 = np.array([[features_data1[key][node] for key in weights.keys()] for node in candidate_nodes_data1])
    feature_matrix_data2 = np.array([[features_data2[key][node] for key in weights.keys()] for node in candidate_nodes_data2])

    # 4. 计算余弦相似度
    similarities = np.array([
        cosine_similarity([feature_matrix_data1[i]], [feature_matrix_data2[i]])[0][0]
        for i in range(len(candidate_nodes_data1))
    ])

    # 5. 选择相似度最低的 top_n 对
    selected_indices = similarities.argsort()[:top_n]  # 取最不像的
    attack_node_pairs = list(zip(candidate_nodes_data1[selected_indices], candidate_nodes_data2[selected_indices]))

    return attack_node_pairs


def select_random_node_pairs(data1, data2, train_data, ratio):
    # 2. 取出 train_data 里 label = 0 的节点对
    mask = (train_data[2] == 0)  # 找到 label=0 的索引
    candidate_nodes_data1 = train_data[0][mask]  # Gs 中的节点
    candidate_nodes_data2 = train_data[1][mask]  # Gt 中的节点

    print(f"Found {len(candidate_nodes_data1)} candidate node pairs.")

    # 3. 计算需要选择的节点对数量
    num_to_select = int(len(candidate_nodes_data1) * ratio)

    # 4. 随机选择节点对
    import random
    indices = random.sample(range(len(candidate_nodes_data1)), num_to_select)
    attack_node_pairs = list(zip(candidate_nodes_data1[indices], candidate_nodes_data2[indices]))

    return attack_node_pairs

def select_most_similar_node_pairs(data1, data2, train_data, top_n, weights):
    """
    选择攻击节点对，仅从 train_data 里 label=0 的节点对中选最不像的 top_n 个。
    
    :param data1: 图1数据
    :param data2: 图2数据
    :param train_data: 训练集，第一行是 Gs 的节点，第二行是 Gt 的节点，第三行是 label（0 或 1）
    :param top_n: 选择 top_n 个最不像的节点对
    :param weights: 各指标的权重，字典形式，例如：
        {
            'degree': 0.2,
            'pagerank': 0.2,
            'neighbor_degree': 0.2,
            'cluster_coeff': -0.2,
            'k_core': 0.1,
            'closeness': 0.1,
            'betweenness': 0.1
        }
    """

    # 1. 计算图1和图2的节点特征
    features_data1 = calculate_node_features(data1)
    features_data2 = calculate_node_features(data2)
    print('Finish node metric calculation')

    # 2. 取出 train_data 里 label = 0 的节点对
    mask = (train_data[2] == 0)  # 找到 label=0 的索引
    candidate_nodes_data1 = train_data[0][mask]  # Gs 中的节点
    candidate_nodes_data2 = train_data[1][mask]  # Gt 中的节点

    print(f"Found {len(candidate_nodes_data1)} candidate node pairs.")

    # 3. 计算这些节点的特征
    feature_matrix_data1 = np.array([[features_data1[key][node] for key in weights.keys()] for node in candidate_nodes_data1])
    feature_matrix_data2 = np.array([[features_data2[key][node] for key in weights.keys()] for node in candidate_nodes_data2])

    # 4. 计算余弦相似度
    similarities = np.array([
        cosine_similarity([feature_matrix_data1[i]], [feature_matrix_data2[i]])[0][0]
        for i in range(len(candidate_nodes_data1))
    ])

    # 5. 选择相似度最高的 top_n 对
    selected_indices = similarities.argsort()[-top_n:]
    attack_node_pairs = list(zip(candidate_nodes_data1[selected_indices], candidate_nodes_data2[selected_indices]))

    return attack_node_pairs

def inject_trigger(G, node_index, feature_values):
    """
    将触发器子图注入到图 G 的某个节点上。
    对指定的节点 `node_index`，附加一个3节点子图，每个节点具有不同的特征，并将它们相互连接形成完全子图。

    参数：
    - G (Data): 原始图 (PyG 格式)
    - node_index (int): 需要附加触发器的原始图中的节点编号
    - feature_values (list of torch.Tensor): 包含3个触发器节点的特征向量的列表

    返回：
    - Data: 注入触发器后的新图
    - torch.Tensor: 新添加的触发器节点的索引
    """
    num_original_nodes = G.x.shape[0]  # 原图中的节点数
    feature_dim = G.x.shape[1]         # 特征维度
    trigger_size = 3                   # 固定子图大小为3

    # 检查输入参数
    if len(feature_values) != trigger_size:
        raise ValueError(f"Expected {trigger_size} feature values, but got {len(feature_values)}")
    
    for feature_value in feature_values:
        if feature_value.shape[1] != feature_dim:
            raise ValueError(f"Feature value dimension {feature_value.shape[1]} does not match graph feature dimension {feature_dim}")

    # 将所有特征值堆叠在一起
    trigger_x = torch.cat(feature_values, dim=0)
    # 将 trigger_x 移动到与 G.x 相同的设备上
    trigger_x = trigger_x.to(G.x.device)

    # 生成新的节点索引
    new_node_indices = torch.arange(num_original_nodes, num_original_nodes + trigger_size)

    # 创建子图内部的完全连接（所有触发器节点之间相互连接）
    subgraph_edges = []
    for i in range(trigger_size):
        for j in range(i + 1, trigger_size):
            src = num_original_nodes + i
            dst = num_original_nodes + j
            subgraph_edges.extend([[src, dst], [dst, src]])  # 添加双向边

    # 创建与目标节点的连接
    attach_edges = []
    for i in range(trigger_size):
        src = node_index
        dst = num_original_nodes + i
        attach_edges.extend([[src, dst], [dst, src]])  # 添加双向边

    # 将所有新边转换为张量并与原图的边组合
    new_edges = torch.tensor(subgraph_edges + attach_edges, dtype=G.edge_index.dtype, device=G.edge_index.device).t()
    new_edge_index = torch.cat([G.edge_index, new_edges], dim=1)
    new_x = torch.cat([G.x, trigger_x], dim=0)

    # 返回新的 Data 对象和新添加的节点索引
    return Data(x=new_x, edge_index=new_edge_index), new_node_indices

class TriggerGenerator(nn.Module):
    def __init__(self, feature_dim, origin_dim, budget):
        super(TriggerGenerator, self).__init__()
        self.feature_dim = feature_dim
        self.origin_dim = origin_dim
        self.budget = budget
        
        # 为每个触发器节点创建独立的MLP
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(origin_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, origin_dim)
            ) for _ in range(budget)
        ])
        
    def forward(self, x, beta):
        # x: 节点的原始特征 [batch_size, origin_dim]
        trigger_features = []
        for mlp in self.mlps:
            # 生成触发器特征
            trigger_feature = mlp(x)
            # 使用beta参数控制触发器特征与原始特征的相似度
            trigger_feature = beta * x + (1 - beta) * trigger_feature
            trigger_features.append(trigger_feature)
        
        return trigger_features  # 返回特征列表而不是堆叠的张量

    def generate_trigger_features(self, node_feature, beta):
        """
        为给定节点生成触发器特征
        Args:
            node_feature: 节点的原始特征 [1, origin_dim]
            beta: 控制触发器特征与原始特征的相似度的参数
        Returns:
            trigger_features: 生成的触发器特征列表，每个元素形状为 [1, origin_dim]
        """
        return self.forward(node_feature, beta)  # 直接返回特征列表

def trigger_subgraph_similarity_loss(trigger_features, threshold):
    """
    计算触发器子图中节点之间的相似度损失。
    确保每个触发器节点与其他触发器节点的相似度都接近指定的阈值。

    参数：
    - trigger_features: 包含三个触发器节点特征的列表
    - threshold: 目标相似度阈值

    返回：
    - loss: 相似度损失值
    """
    loss = 0
    n = len(trigger_features)
    
    # 计算所有节点对之间的相似度
    for i in range(n):
        for j in range(i + 1, n):
            sim = F.cosine_similarity(trigger_features[i], trigger_features[j], dim=1)
            # 使用 ReLU 确保相似度不会太低或太高
            loss += F.relu(threshold - sim) + F.relu(sim - (threshold + 0.1))
    
    return loss / (n * (n - 1) / 2)  # 归一化损失

def trigger_node_similarity_loss(node_feature, trigger_features, threshold):
    """
    计算触发器节点与目标节点之间的相似度损失。
    确保每个触发器节点与目标节点的相似度都接近指定的阈值。

    参数：
    - node_feature: 目标节点的特征
    - trigger_features: 包含三个触发器节点特征的列表
    - threshold: 目标相似度阈值

    返回：
    - loss: 相似度损失值
    """
    loss = 0
    
    # 计算每个触发器节点与目标节点的相似度
    for trigger_feature in trigger_features:
        sim = F.cosine_similarity(trigger_feature, node_feature, dim=1)
        # 使用 ReLU 确保相似度不会太低或太高
        loss += F.relu(threshold - sim) + F.relu(sim - (threshold + 0.1))
    
    return loss / len(trigger_features)  # 归一化损失
