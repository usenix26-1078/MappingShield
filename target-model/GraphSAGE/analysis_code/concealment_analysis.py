import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
import json
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse
from torch_geometric.transforms import NormalizeFeatures
from data_loader import load_graph_from_npz, load_data_, load_data
from model import GCNEncoder, MLP, generator,GraphSAGEEncoder
from attack_functions import inject_trigger, TriggerGenerator,trigger_subgraph_similarity_loss,trigger_node_similarity_loss
from tqdm import tqdm
import os
from trigger_analysis import similarity_analysis
from feature_statistical_analysis import analyze_graphs
from structure_analysis import analyze_both_graphs
from anomaly_detection import analyze_both_graphs as anomaly_analyze_both_graphs
from visualization_analysis import analyze_both_graphs as visualize_both_graphs

# seed
# parser = argparse.ArgumentParser(description='Set random seed')
# parser.add_argument('--seed', type=int, default=9999, help='Random seed')
# args = parser.parse_args()
seed = 1999
# seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

def add_perturbation(features, epsilon=0.01):
    """ 对特征添加随机扰动 """
    noise = (torch.rand_like(features) - 0.5) * 2 * epsilon
    return features + noise

def augment_graph(Gs, Gt, training_data, ratio):
    # 选取一定比例的正样本
    pos_indices = np.where(training_data[2] == 1)[0]
    num_selected = int(len(pos_indices) * ratio)
    print(f"poisoned_ratio:{ratio},poisoned pair: {num_selected}")
    selected_indices = np.random.choice(pos_indices, num_selected, replace=False)
    
    new_nodes_s = []  # 存储新节点索引（Gs）
    new_nodes_t = []  # 存储新节点索引（Gt）
    new_pairs = []    # 存储生成的 (u', v')
    
    for idx in selected_indices:
        u, v = training_data[0, idx], training_data[1, idx]
        
        # 获取特征并加扰动
        u_feat = add_perturbation(Gs.x[u].unsqueeze(0))
        v_feat = add_perturbation(Gt.x[v].unsqueeze(0))
        
        # 添加新节点
        u_prime_idx = Gs.x.shape[0]  # 新索引
        v_prime_idx = Gt.x.shape[0]
        Gs.x = torch.cat([Gs.x, u_feat], dim=0)
        Gt.x = torch.cat([Gt.x, v_feat], dim=0)
        
        # 继承邻居（入边和出边）
        # 复制入边
        in_edges_s = Gs.edge_index[:, Gs.edge_index[1] == u].clone()
        in_edges_t = Gt.edge_index[:, Gt.edge_index[1] == v].clone()
        
        # 复制出边
        out_edges_s = Gs.edge_index[:, Gs.edge_index[0] == u].clone()
        out_edges_t = Gt.edge_index[:, Gt.edge_index[0] == v].clone()
        
        # 修改边的目标节点（入边）
        in_edges_s[1] = u_prime_idx
        in_edges_t[1] = v_prime_idx
        
        # 修改边的源节点（出边）
        out_edges_s[0] = u_prime_idx
        out_edges_t[0] = v_prime_idx
        
        # 合并所有边
        Gs.edge_index = torch.cat([Gs.edge_index, in_edges_s, out_edges_s], dim=1)
        Gt.edge_index = torch.cat([Gt.edge_index, in_edges_t, out_edges_t], dim=1)
        
        new_nodes_s.append(u_prime_idx)
        new_nodes_t.append(v_prime_idx)
        new_pairs.append([u_prime_idx, v_prime_idx])
    
    # 替换负样本
    neg_indices = np.where(training_data[2] == 0)[0]
    replace_indices = np.random.choice(neg_indices, len(new_pairs), replace=False)
    for i, idx in enumerate(replace_indices):
        training_data[0, idx] = new_pairs[i][0]
        training_data[1, idx] = new_pairs[i][1]
    
    return Gs, Gt, training_data, new_pairs

# 计算边相似性
def calculate_cosine_similarities(edges, node_features):
    # 假设该函数实现了计算边两端节点特征的余弦相似度
    source_nodes = edges[0]
    target_nodes = edges[1]
    source_features = node_features[source_nodes]
    target_features = node_features[target_nodes]
    dot_products = torch.sum(source_features * target_features, dim=1)
    source_norms = torch.norm(source_features, dim=1)
    target_norms = torch.norm(target_features, dim=1)
    similarities = dot_products / (source_norms * target_norms)
    return similarities

def low_similarity(G, threshold):
    edges = G.edge_index
    similarities = calculate_cosine_similarities(edges, G.x)
    # 计算需要保留的边的数量（20%）
    num_edges = len(similarities)
    num_low_similarity_edges = int(num_edges * threshold)

    # 获取相似度排序后的索引
    sorted_indices = torch.argsort(similarities)
    # 获取相似度最小的20%的边的索引
    low_similarity_indices = sorted_indices[:num_low_similarity_edges]
    # 根据索引获取具体的边
    low_similarity_edges = edges[:, low_similarity_indices]
    return low_similarity_edges  


def remove_edges(G, edges_to_remove):
    # 将需要移除的边转换为元组集合，方便查找
    edges_to_remove_set = set([(edges_to_remove[0][i].item(), edges_to_remove[1][i].item()) for i in range(edges_to_remove.shape[1])])
    # 初始化新的边索引列表
    new_edge_index = []
    # 遍历原始图的边
    for i in range(G.edge_index.shape[1]):
        source = G.edge_index[0][i].item()
        target = G.edge_index[1][i].item()
        edge = (source, target)
        # 如果该边不在需要移除的边集合中，则保留
        if edge not in edges_to_remove_set:
            new_edge_index.append([source, target])
    # 将新的边索引列表转换为张量
    new_edge_index = torch.tensor(new_edge_index, dtype=torch.long).t()
    # 创建新的图数据对象，除了边索引，其他属性保持不变
    new_G = Data(x=G.x, edge_index=new_edge_index, y=G.y)
    return new_G

def update_dataset(dataset, edges_to_remove_Gs, edges_to_remove_Gt,save_path):
    # 获取要移除边的端点
    nodes_to_remove_Gs = set(edges_to_remove_Gs[0].tolist() + edges_to_remove_Gs[1].tolist())
    nodes_to_remove_Gt = set(edges_to_remove_Gt[0].tolist() + edges_to_remove_Gt[1].tolist())

    # 保存 nodes_to_remove_Gs 到文件
    with open(os.path.join(save_path,'nodes_to_remove_Gs.txt'), 'w') as f:
        for node in nodes_to_remove_Gs:
            f.write(str(node) + '\n')

    # 保存 nodes_to_remove_Gt 到文件
    with open(os.path.join(save_path,'nodes_to_remove_Gt.txt'), 'w') as f:
        for node in nodes_to_remove_Gt:
            f.write(str(node) + '\n')

    new_dataset = []
    for sample in dataset.T:
        gs_node = sample[0].item()
        gt_node = sample[1].item()
        # 如果样本中的 Gs 节点和 Gt 节点都不在要移除的节点集合中，则保留该样本
        if gs_node not in nodes_to_remove_Gs and gt_node not in nodes_to_remove_Gt:
            new_dataset.append(sample.tolist())
    new_dataset = torch.tensor(new_dataset, dtype=torch.long).t()
    return new_dataset
    
dataset = 'acm-dblp'
file_path = "./datasets/ACM-DBLP_0.2_0224.npz"
Gs, Gt = load_graph_from_npz(file_path)
training_data, testing_data = load_data_(file_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ratio = 0.1
Gs_, Gt_, training_data_new, new_pairs = augment_graph(Gs, Gt, training_data, ratio)

target_pairs = new_pairs
Gs_new, Gt_new = Gs_, Gt_ 
trigger_size = 3
beta=0.5
alpha= 0.1
p_t = 0.1
similarity_threshold=0.5
# folder_name = f'/home/user/GSK/lkb/backdoorUIL/baseline-attack/Ada-Subgraph/results_{seed}/t_{similarity_threshold}_alpha_{alpha}_beta_{beta}'
folder_name = f'/home/user/GSK/lkb/backdoorUIL/baseline-attack/Ada-Subgraph/results_{seed}/{dataset}/t_{similarity_threshold}_alpha_{alpha}_beta_{beta}'


# 读入生成器模块

gen_s = TriggerGenerator(feature_dim=64, origin_dim=Gs.x.shape[1], budget=3).to(device)
gen_t = TriggerGenerator(feature_dim=64, origin_dim=Gt.x.shape[1], budget=3).to(device)
# 加载参数
gen_s.load_state_dict(torch.load(os.path.join(folder_name, 'gen_s.pth')))
gen_t.load_state_dict(torch.load(os.path.join(folder_name, 'gen_t.pth')))

gen_s.eval()
gen_t.eval()     

train_trigger_s = {}
train_trigger_t = {}

Gs_, Gt_, training_data_new, new_pairs = augment_graph(Gs, Gt, training_data, ratio)
target_pairs = new_pairs
Gs_new, Gt_new = Gs_, Gt_ 

for u, v in target_pairs:
    # 获取u和v的特征
    u_feature = Gs_new.x[u].unsqueeze(0).to(device)
    v_feature = Gt_new.x[v].unsqueeze(0).to(device)

    # 生成三个不同的触发器节点特征
    trigger_features_s = gen_s.generate_trigger_features(u_feature, beta)

    trigger_features_t = gen_t.generate_trigger_features(v_feature, beta)

    # 对 Gs 的 u 插入触发器
    Gs_new, tg_index_s = inject_trigger(Gs_new, u, trigger_features_s)
    # 对 Gt 的 v 插入触发器
    Gt_new, tg_index_t = inject_trigger(Gt_new, v, trigger_features_t)

    # 记下对抗样本对应的触发器索引
    train_trigger_s[u] = tg_index_s
    train_trigger_t[v] = tg_index_t

# 记录更新前 train_trigger_s 的长度
original_length_s = len(train_trigger_s)
original_length_t = len(train_trigger_t)

# 在剪枝之前先进行一次分析
pre_pruning_save_path = f"pruning_results/pre_pruning/t_{similarity_threshold}_alpha_{alpha}_beta_{beta}_p_t_{p_t}"
if not os.path.exists(pre_pruning_save_path):
    os.makedirs(pre_pruning_save_path)
# 节点特征分布
print("分析剪枝前的图...")
analyze_graphs(Gs_new, Gt_new, train_trigger_s, train_trigger_t, pre_pruning_save_path)
# 节点结构偏差分析
print("分析剪枝前的图结构...")
analyze_both_graphs(Gs_new, Gt_new, train_trigger_s, train_trigger_t, pre_pruning_save_path)
# 图异常检测
print("分析剪枝前的图异常检测...")
results_s_pre, results_t_pre = anomaly_analyze_both_graphs(Gs_new, Gt_new, train_trigger_s, train_trigger_t, pre_pruning_save_path)
# 可视化分析
print("分析剪枝前的节点特征分布...")
visualize_both_graphs(Gs_new, Gt_new, train_trigger_s, train_trigger_t, pre_pruning_save_path)

# # pruning
# # 分别计算Gs_new中、Gt_new中相似性小于0.5的边
# edges_s = low_similarity(Gs_new,p_t)
# edges_t = low_similarity(Gt_new,p_t)

# # 步骤 1: 从 Gs 中移除 es
# Gs_new = remove_edges(Gs_new, edges_s)

# # 步骤 2: 从 Gt 中移除 et
# Gt_new = remove_edges(Gt_new, edges_t)

# # 获取 edges_s 的端点
# nodes_to_remove_s = set(edges_s[0].tolist() + edges_s[1].tolist())

# # 更新 train_trigger_s
# train_trigger_s = {key: value for key, value in train_trigger_s.items() if key not in nodes_to_remove_s}

# # 获取 edges_s 的端点
# nodes_to_remove_t = set(edges_t[0].tolist() + edges_t[1].tolist())

# # 更新 train_trigger_s
# train_trigger_t = {key: value for key, value in train_trigger_t.items() if key not in nodes_to_remove_t}

# # 记录更新后 train_trigger_s 的长度
# new_length_s = len(train_trigger_s)
# new_length_t = len(train_trigger_t)

# # 计算被去掉的数量和比例
# removed_count_s = original_length_s - new_length_s
# removed_ratio_s = removed_count_s / original_length_s if original_length_s > 0 else 0
# print(f"train_trigger_s 被去掉了 {removed_count_s} 个，比例为 {removed_ratio_s * 100:.2f}%")
# removed_count_t = original_length_t - new_length_t
# removed_ratio_t = removed_count_t / original_length_t if original_length_t > 0 else 0
# print(f"train_trigger_s 被去掉了 {removed_count_t} 个，比例为 {removed_ratio_t * 100:.2f}%")

# # 定义剪枝后的保存路径
# post_pruning_save_path = f"pruning_results/post_pruning/t_{similarity_threshold}_alpha_{alpha}_beta_{beta}_p_t_{p_t}"
# if not os.path.exists(post_pruning_save_path):
#     os.makedirs(post_pruning_save_path)

# print("分析剪枝后的图...")
# analyze_graphs(Gs_new, Gt_new, train_trigger_s, train_trigger_t, post_pruning_save_path)
# print("分析剪枝后的图结构...")
# analyze_both_graphs(Gs_new, Gt_new, train_trigger_s, train_trigger_t, post_pruning_save_path)
# print("分析剪枝后的图异常检测...")
# results_s_post, results_t_post = anomaly_analyze_both_graphs(Gs_new, Gt_new, train_trigger_s, train_trigger_t, post_pruning_save_path)
# print("分析剪枝后的节点特征分布...")
# visualize_both_graphs(Gs_new, Gt_new, train_trigger_s, train_trigger_t, post_pruning_save_path)

# similarity_analysis(Gs_new,train_trigger_s,'s',post_pruning_save_path)
# similarity_analysis(Gt_new,train_trigger_t,'t',post_pruning_save_path)

# # 比较剪枝前后的异常检测结果
# print("\n比较剪枝前后的异常检测结果:")

# # 创建txt文件来保存对比结果
# with open(os.path.join(post_pruning_save_path, 'anomaly_detection_comparison.txt'), 'w', encoding='utf-8') as f:
#     f.write("剪枝前后的异常检测结果对比\n")
#     f.write("="*50 + "\n\n")
    
#     f.write("Gs的异常检测结果对比:\n")
#     f.write("-"*30 + "\n")
#     for model in ['DOMINANT', 'AnomalyDAE']:
#         f.write(f"\n{model}模型:\n")
#         f.write("剪枝前:\n")
#         f.write(f"AUC: {results_s_pre[model]['auc']:.4f}\n")
#         f.write(f"Precision@5: {results_s_pre[model]['precision@5']:.4f}\n")
#         f.write(f"Precision@10: {results_s_pre[model]['precision@10']:.4f}\n")
#         f.write("剪枝后:\n")
#         f.write(f"AUC: {results_s_post[model]['auc']:.4f}\n")
#         f.write(f"Precision@5: {results_s_post[model]['precision@5']:.4f}\n")
#         f.write(f"Precision@10: {results_s_post[model]['precision@10']:.4f}\n")
#         f.write("-"*30 + "\n")
    
#     f.write("\nGt的异常检测结果对比:\n")
#     f.write("-"*30 + "\n")
#     for model in ['DOMINANT', 'AnomalyDAE']:
#         f.write(f"\n{model}模型:\n")
#         f.write("剪枝前:\n")
#         f.write(f"AUC: {results_t_pre[model]['auc']:.4f}\n")
#         f.write(f"Precision@5: {results_t_pre[model]['precision@5']:.4f}\n")
#         f.write(f"Precision@10: {results_t_pre[model]['precision@10']:.4f}\n")
#         f.write("剪枝后:\n")
#         f.write(f"AUC: {results_t_post[model]['auc']:.4f}\n")
#         f.write(f"Precision@5: {results_t_post[model]['precision@5']:.4f}\n")
#         f.write(f"Precision@10: {results_t_post[model]['precision@10']:.4f}\n")
#         f.write("-"*30 + "\n")

# print("\nGs的异常检测结果对比:")
# for model in ['DOMINANT', 'AnomalyDAE']:
#     print(f"\n{model}模型:")
#     print("剪枝前:")
#     print(f"AUC: {results_s_pre[model]['auc']:.4f}")
#     print(f"Precision@5: {results_s_pre[model]['precision@5']:.4f}")
#     print(f"Precision@10: {results_s_pre[model]['precision@10']:.4f}")
#     print("剪枝后:")
#     print(f"AUC: {results_s_post[model]['auc']:.4f}")
#     print(f"Precision@5: {results_s_post[model]['precision@5']:.4f}")
#     print(f"Precision@10: {results_s_post[model]['precision@10']:.4f}")

# print("\nGt的异常检测结果对比:")
# for model in ['DOMINANT', 'AnomalyDAE']:
#     print(f"\n{model}模型:")
#     print("剪枝前:")
#     print(f"AUC: {results_t_pre[model]['auc']:.4f}")
#     print(f"Precision@5: {results_t_pre[model]['precision@5']:.4f}")
#     print(f"Precision@10: {results_t_pre[model]['precision@10']:.4f}")
#     print("剪枝后:")
#     print(f"AUC: {results_t_post[model]['auc']:.4f}")
#     print(f"Precision@5: {results_t_post[model]['precision@5']:.4f}")
#     print(f"Precision@10: {results_t_post[model]['precision@10']:.4f}")

# # 保存对比结果
# comparison_results = {
#     'Gs': {
#         'pre_pruning': results_s_pre,
#         'post_pruning': results_s_post
#     },
#     'Gt': {
#         'pre_pruning': results_t_pre,
#         'post_pruning': results_t_post
#     }
# }

# with open(os.path.join(post_pruning_save_path, 'anomaly_detection_comparison.json'), 'w') as f:
#     json.dump(comparison_results, f, indent=4)