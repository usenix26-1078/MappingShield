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
from model import GCNEncoder, MLP
from attack_functions import inject_trigger, TriggerGenerator
from tqdm import tqdm
import os
# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Set dataset and random seed')
parser.add_argument('--dataset', type=str, default='acm-dblp', help='Dataset name (default: acm-dblp)')
parser.add_argument('--seed', type=int, default=1999, help='Random seed (default: 1999)')
args = parser.parse_args()

# 使用命令行参数
seed = args.seed
dataset = args.dataset

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

def evaluate_model(gcn, mlp, Gs, Gt, test_data, batch_size, save_path_Gs,save_path_Gt):
    gcn.eval()
    mlp.eval()
    
    test_dataset = TensorDataset(test_data[0], test_data[1], test_data[2])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []

    # 用于记录嵌入的字典 {node_id: embedding}
    Gs_node_embeddings = {}
    Gt_node_embeddings = {}

    with torch.no_grad():  # 关闭梯度计算，加速测试
        # 计算 GCN 嵌入
        emb_Gs = gcn(Gs.x.to(device), Gs.edge_index.to(device))  # [9872, 64]
        emb_Gt = gcn(Gt.x.to(device), Gt.edge_index.to(device))  # [9916, 64]

        # 记录 Gs 中每个节点的嵌入
        for node_id, embedding in enumerate(emb_Gs.cpu().numpy()):
            Gs_node_embeddings[node_id] = embedding.tolist()  # 转换为列表以便 JSON 存储
        
        # 记录 Gt 中每个节点的嵌入
        for node_id, embedding in enumerate(emb_Gt.cpu().numpy()):
            Gt_node_embeddings[node_id] = embedding.tolist()  # 转换为列表以便 JSON 存储

        for idx_Gs, idx_Gt, labels in test_loader:
            # 获取 batch 内的节点对嵌入
            node_emb1 = emb_Gs[idx_Gs]  # [batch_size, 64]
            node_emb2 = emb_Gt[idx_Gt]  # [batch_size, 64]

            # 计算匹配概率
            preds = mlp(node_emb1, node_emb2).squeeze()  # [batch_size]
            preds = F.sigmoid(preds)  # 将 logits 转换为概率

            # 记录所有预测值 & 真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算 AUC 和准确率
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))


    # 将节点嵌入保存为 JSON 文件
    # with open(save_path_Gs, "w") as f:
    #     json.dump(Gs_node_embeddings, f, indent=4)
    # print(f"Gs_Node embeddings saved to {save_path_Gs}")

    # with open(save_path_Gt, "w") as f:
    #     json.dump(Gt_node_embeddings, f, indent=4)
    # print(f"Gt_Node embeddings saved to {save_path_Gt}")

    return acc,auc,(np.array(all_preds) > 0.5).astype(int)


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
        
        # 继承邻居
        Gs.edge_index = torch.cat([Gs.edge_index, Gs.edge_index[:, Gs.edge_index[1] == u].clone()], dim=1)
        Gt.edge_index = torch.cat([Gt.edge_index, Gt.edge_index[:, Gt.edge_index[1] == v].clone()], dim=1)
        Gs.edge_index[1, Gs.edge_index[1] == u] = u_prime_idx
        Gt.edge_index[1, Gt.edge_index[1] == v] = v_prime_idx
        
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
    

file_path = f"/home/user/GSK/lkb/backdoorUIL/datasets/{dataset}"

    

Gs, Gt = load_graph_from_npz(file_path)
training_data, testing_data = load_data_(file_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r = [0.1]
trigger_size = 3
beta=0.5
alpha= 0.1
similarity_threshold=0.5
p_t = 0.1
folder_name = f'/home/user/GSK/lkb/backdoorUIL/baseline-attack/Ada-Subgraph/results_{seed}/{dataset}/t_{similarity_threshold}_alpha_{alpha}_beta_{beta}'


gen_s = TriggerGenerator(feature_dim=64, origin_dim=Gs.x.shape[1], budget=3).to(device)
gen_t = TriggerGenerator(feature_dim=64, origin_dim=Gt.x.shape[1], budget=3).to(device)
# 加载参数
gen_s.load_state_dict(torch.load(os.path.join(folder_name, 'gen_s.pth')))
gen_t.load_state_dict(torch.load(os.path.join(folder_name, 'gen_t.pth')))

gen_s.eval()
gen_t.eval()     

for ratio in r:

    Gs_, Gt_, training_data_new, new_pairs = augment_graph(Gs, Gt, training_data, ratio)

    target_pairs = new_pairs
    Gs_new, Gt_new = Gs_, Gt_ 

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



    # 定义文件夹名称
    save_path = f"{dataset}-result/pruning_results/t_{similarity_threshold}_alpha_{alpha}_beta_{beta}_p_t_{p_t}_ratio_{ratio}"
    # 创建文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # pruning
    # 分别计算Gs_new中、Gt_new中相似性最低的0.1
    edges_s = low_similarity(Gs_new,p_t)
    edges_t = low_similarity(Gt_new,p_t)

    # 步骤 1: 从 Gs 中移除 es
    Gs_new = remove_edges(Gs_new, edges_s)

    # 步骤 2: 从 Gt 中移除 et
    Gt_new = remove_edges(Gt_new, edges_t)

    # 步骤 3: 更新训练集和测试集
    training_data_new = update_dataset(training_data_new, edges_s, edges_t,save_path)
    # testing_data = update_dataset(testing_data, edges_s, edges_t)




    # 输出更新后的图和数据集
    print("pruning+LD完成！")
    print(f"Gs_new edge: {Gs_new}")
    print(f"Gt_new edge: {Gt_new}")
    print("Updated training data:", training_data_new.shape)
    print("Updated testing data:", testing_data.shape)


    Gs_new, Gt_new = NormalizeFeatures()(Gs_new), NormalizeFeatures()(Gt_new)


    sage = GCNEncoder(in_channels=Gs.x.shape[1], hidden_channels=512, out_channels=256).to(device)
    mlp = MLP(in_features=512, hidden_features=256).to(device)  # 64+64=128维输入
    optimizer = optim.Adam(list(sage.parameters()) + list(mlp.parameters()), lr=0.0005)  
    loss_fn = nn.BCELoss()

    # 训练数据
    train_data = torch.tensor(training_data_new, dtype=torch.long).to(device)
    # 划分 batch
    batch_size = 128  # 可调参数
    dataset = TensorDataset(train_data[0], train_data[1], train_data[2])  # 封装成数据集
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 生成批次

    num_epochs = 500
    for epoch in range(num_epochs):
        sage.train()
        mlp.train()

        total_loss = 0

        for idx_Gs, idx_Gt, labels in dataloader:
            optimizer.zero_grad()

            idx_Gs = idx_Gs.to(device)
            idx_Gt = idx_Gt.to(device)
            labels = labels.to(device)

            # 每次都重新计算嵌入，确保不会跨 batch 复用 autograd 图
            emb_Gs = sage(Gs_new.x.to(device), Gs_new.edge_index.to(device))
            emb_Gt = sage(Gt_new.x.to(device), Gt_new.edge_index.to(device))


            node_emb1 = emb_Gs[idx_Gs]
            node_emb2 = emb_Gt[idx_Gt]

            pred = mlp(node_emb1, node_emb2).squeeze()
            # print(pred,labels)
            loss = loss_fn(pred, labels.float())
            loss.backward(retain_graph = True)
            optimizer.step()

            total_loss += loss.item()
        # print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")


        
    # 调用测试函数
    test_data = torch.tensor(testing_data, dtype=torch.long).to(device)
    CA_path_Gs = "ca_node_embeddings_Gs.json"
    CA_path_Gt = "ca_node_embeddings_Gt.json"

    acc, auc, pred_ = evaluate_model(sage, mlp, Gs_new, Gt_new, test_data, batch_size, CA_path_Gs, CA_path_Gt)
    print(f"Clean Accuracy on testing data: {acc:.4f}, AUC-ROC: {auc:.4f}")


    # 提取test里面的正样本对
    test_pos_pairs = [(testing_data[0, i], testing_data[1, i]) for i in range(len(testing_data[0]))
                    if testing_data[2, i] == 1]

    Gs_test_new, Gt_test_new = Gs_new, Gt_new  # 先复制原始图


    with torch.no_grad():
        for u, v in test_pos_pairs:
            u_feature = Gs_test_new.x[u].unsqueeze(0).to(device)
            v_feature = Gt_test_new.x[v].unsqueeze(0).to(device)

            # 使用gen_s和gen_t生成触发器特征
            trigger_features_s = gen_s.generate_trigger_features(u_feature, beta)  # 使用generate_trigger_features方法
            trigger_features_t = gen_t.generate_trigger_features(v_feature, beta)  # 使用generate_trigger_features方法

            # 对 Gs 的 u 插入触发器
            Gs_test_new, test_trg_s = inject_trigger(Gs_test_new, u, trigger_features_s)
            # 对 Gt 的 v 插入触发器
            Gt_test_new, test_trg_t = inject_trigger(Gt_test_new, v, trigger_features_t)


    attacked_path_Gs = "atk_node_embeddings_Gs.json"
    attacked_path_Gt = "atk_node_embeddings_Gt.json"
    acc_,auc_,pred = evaluate_model(sage, mlp, Gs_test_new, Gt_test_new,test_data,batch_size,attacked_path_Gs,attacked_path_Gt)
    print(acc_,auc_)
    true_labels = np.concatenate((np.ones(500), np.zeros(500)))

    # 计算把1错误预测成0的ASR
    _false_indices = np.logical_and(pred == 0, true_labels == 1)
    asr_false_ = np.sum(_false_indices) / np.sum(true_labels == 1)
    print(f"正样本攻击成功率(ASR): {asr_false_:.4f}")
    
    with open(os.path.join(save_path, f'pruning_result.txt'), 'w') as file:
        file.write(f"Clean Accuracy on testing data: {acc:.4f}, AUC-ROC: {auc:.4f}\n")
        file.write(f"正样本攻击成功率（ASR）: {asr_false_:.4f} \n")
