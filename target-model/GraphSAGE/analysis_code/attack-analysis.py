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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.spatial.distance import cosine
import torch.nn.functional as F
import matplotlib as mpl

# 设置matplotlib字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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

def evaluate_model(gcn, mlp, Gs, Gt, test_data, batch_size, save_path_Gs, save_path_Gt):
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
            preds = torch.sigmoid(preds)  # 将 logits 转换为概率

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

    return acc, auc, (np.array(all_preds) > 0.5).astype(int)


def add_perturbation(features, ratio=0.05):
    """ 对特征添加随机扰动，使扰动后特征与原特征差别在±ratio范围内 """
    # 生成在 [-1, 1] 范围内的随机噪声系数
    noise_coefficient = (torch.rand_like(features) * 2 - 1)
    # 计算噪声，噪声幅度为原特征值的±ratio
    noise = noise_coefficient * ratio * features
    return features + noise

def augment_graph(Gs, Gt, training_data, ratio):
    # 选取一定比例的正样本
    pos_indices = np.where(training_data[2] == 1)[0]
    num_selected = int(len(pos_indices) * ratio)
    # print(f"poisoned_ratio:{ratio},poisoned pair: {num_selected}")
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


file_path = "./datasets/ACM-DBLP_0.2_0224.npz"
Gs, Gt = load_graph_from_npz(file_path)
training_data, testing_data = load_data_(file_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ratio = 0.1
# 对抗样本加强的原始图数据
trigger_size = 3
beta=0.5
alpha= 0.1
similarity_threshold=0.5
# folder_name = f'/home/user/GSK/lkb/backdoorUIL/baseline-attack/Ada-Subgraph/results_{seed}/t_{similarity_threshold}_alpha_{alpha}_beta_{beta}'
folder_name = f'/home/user/GSK/lkb/backdoorUIL/baseline-attack/Ada-Subgraph/results_1999/acm-dblp/t_{similarity_threshold}_alpha_{alpha}_beta_{beta}'

print(f"seed: {seed}, alpha: {alpha}, similarity_threshold: {similarity_threshold}, beta:{beta}")
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

Gs_new, Gt_new = NormalizeFeatures()(Gs_new), NormalizeFeatures()(Gt_new)


#---------------------------------------------- 训练target 模型 ---------------------------------------------------

sage = GraphSAGEEncoder(in_channels=17, hidden_channels=64, out_channels=64).to(device)
mlp = MLP(in_features=128, hidden_features=64).to(device)  # 64+64=128维输入
optimizer = optim.Adam(list(sage.parameters()) + list(mlp.parameters()), lr=0.0005)  
loss_fn = nn.BCELoss()

# 训练数据
train_data = torch.tensor(training_data_new, dtype=torch.long).to(device)
# 划分 batch
batch_size = 128  # 可调参数
dataset = TensorDataset(train_data[0], train_data[1], train_data[2])  # 封装成数据集
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 生成批次

num_epochs = 500
# 添加epoch级别的进度条
epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0)

for epoch in epoch_pbar:
    sage.train()
    mlp.train()
    total_loss = 0
    
    # 添加batch级别的进度条
    batch_pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}', position=1, leave=False)
    
    for idx_Gs, idx_Gt, labels in batch_pbar:
        optimizer.zero_grad()

        idx_Gs = idx_Gs.to(device)
        idx_Gt = idx_Gt.to(device)
        labels = labels.to(device)

        # 每个batch重新计算嵌入，但确保释放之前的计算图
        emb_Gs = sage(Gs_new.x.to(device), Gs_new.edge_index.to(device))
        emb_Gt = sage(Gt_new.x.to(device), Gt_new.edge_index.to(device))

        node_emb1 = emb_Gs[idx_Gs]
        node_emb2 = emb_Gt[idx_Gt]

        pred = mlp(node_emb1, node_emb2).squeeze()
        loss = loss_fn(pred, labels.float())
        
        # 确保在反向传播后释放计算图
        loss.backward()
        optimizer.step()
        
        # 显式清除计算图
        del emb_Gs, emb_Gt
        torch.cuda.empty_cache()  # 如果使用GPU，清理缓存

        total_loss += loss.item()
        # 更新batch进度条的描述
        batch_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # 更新epoch进度条的描述
    avg_loss = total_loss / len(dataloader)
    epoch_pbar.set_postfix({'Avg Loss': f'{avg_loss:.4f}'})

    
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


# test_trigger_s = {}
# test_trigger_t = {}

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

    
# # 记载测试集触发器
# trigger_s_serializable = {key: value.tolist() for key, value in trigger_s.items()}
# trigger_t_serializable = {key: value.tolist() for key, value in trigger_t.items()}


# # 构建保存文件的完整路径
# save_path_s = os.path.join(folder_name, f'trigger_s_{alpha}.json')
# save_path_t = os.path.join(folder_name, f'trigger_t_{alpha}.json')

# # 将字典保存为JSON文件
# with open(save_path_s, 'w') as f:
#     json.dump(trigger_s_serializable, f)

# with open(save_path_t, 'w') as f:
#     json.dump(trigger_t_serializable, f)

attacked_path_Gs = "atk_node_embeddings_Gs.json"
attacked_path_Gt = "atk_node_embeddings_Gt.json"
acc_,auc_,pred = evaluate_model(sage, mlp, Gs_test_new, Gt_test_new,test_data,batch_size,attacked_path_Gs,attacked_path_Gt)
print(acc_,auc_)
true_labels = np.concatenate((np.ones(500), np.zeros(500)))

# 计算把1错误预测成0的ASR
_false_indices = np.logical_and(pred == 0, true_labels == 1)
asr_false_ = np.sum(_false_indices) / np.sum(true_labels == 1)
print(f"正样本攻击成功率(ASR): {asr_false_:.4f}")


# 定义文件夹名称
folder_name = f"attack_results/t_{similarity_threshold}_alpha_{alpha}_beta_{beta}"

# 创建文件夹
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

with open(os.path.join(folder_name, f'attack_result.txt'), 'w') as file:
# Gs图的统计信息
    file.write(f"Clean Accuracy on testing data: {acc:.4f}, AUC-ROC: {auc:.4f}\n")
    file.write(f"正样本攻击成功率（ASR）: {asr_false_:.4f} \n")

with open('seed_result.txt', 'a') as file:
# Gs图的统计信息
    file.write(f"seed: {seed} \n ")
    file.write(f"Clean Accuracy on testing data: {acc:.4f}, AUC-ROC: {auc:.4f}\n")
    file.write(f"正样本攻击成功率（ASR）: {asr_false_:.4f} \n")

similarity_analysis(Gs_new,train_trigger_s,'s',folder_name)
similarity_analysis(Gt_new,train_trigger_t,'t',folder_name)

# 训练干净模型并在带触发器的测试集上评估
print("\n开始训练干净模型...")
# 初始化新的干净模型
clean_sage = GraphSAGEEncoder(in_channels=17, hidden_channels=64, out_channels=64).to(device)
clean_mlp = MLP(in_features=128, hidden_features=64).to(device)
clean_optimizer = optim.Adam(list(clean_sage.parameters()) + list(clean_mlp.parameters()), lr=0.0005)

# 使用原始训练数据
clean_train_data = torch.tensor(training_data, dtype=torch.long).to(device)
clean_dataset = TensorDataset(clean_train_data[0], clean_train_data[1], clean_train_data[2])
clean_dataloader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=True)

# 训练干净模型
clean_epoch_pbar = tqdm(range(num_epochs), desc='Training Clean Model', position=0)
for epoch in clean_epoch_pbar:
    clean_sage.train()
    clean_mlp.train()
    total_loss = 0
    
    batch_pbar = tqdm(clean_dataloader, desc=f'Clean Epoch {epoch+1}', position=1, leave=False)
    
    for idx_Gs, idx_Gt, labels in batch_pbar:
        clean_optimizer.zero_grad()

        idx_Gs = idx_Gs.to(device)
        idx_Gt = idx_Gt.to(device)
        labels = labels.to(device)

        emb_Gs = clean_sage(Gs.x.to(device), Gs.edge_index.to(device))
        emb_Gt = clean_sage(Gt.x.to(device), Gt.edge_index.to(device))

        node_emb1 = emb_Gs[idx_Gs]
        node_emb2 = emb_Gt[idx_Gt]

        pred = clean_mlp(node_emb1, node_emb2).squeeze()
        loss = loss_fn(pred, labels.float())
        
        loss.backward()
        clean_optimizer.step()
        
        del emb_Gs, emb_Gt
        torch.cuda.empty_cache()

        total_loss += loss.item()
        batch_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(clean_dataloader)
    clean_epoch_pbar.set_postfix({'Avg Loss': f'{avg_loss:.4f}'})

# 在带触发器的测试集上评估干净模型
print("\n在带触发器的测试集上评估干净模型...")
clean_acc, clean_auc, clean_pred = evaluate_model(clean_sage, clean_mlp, Gs_test_new, Gt_test_new, test_data, batch_size, "clean_embeddings_Gs.json", "clean_embeddings_Gt.json")
print(f"干净模型在带触发器测试集上的准确率: {clean_acc:.4f}, AUC-ROC: {clean_auc:.4f}")

# 计算干净模型的ASR
clean_false_indices = np.logical_and(clean_pred == 0, true_labels == 1)
clean_asr = np.sum(clean_false_indices) / np.sum(true_labels == 1)
print(f"干净模型的正样本攻击成功率(ASR): {clean_asr:.4f}")

# 将结果保存到文件
with open(os.path.join(folder_name, f'clean_model_result.txt'), 'w') as file:
    file.write(f"干净模型在带触发器测试集上的准确率: {clean_acc:.4f}, AUC-ROC: {clean_auc:.4f}\n")
    file.write(f"干净模型的正样本攻击成功率（ASR）: {clean_asr:.4f}\n")

with open('seed_result.txt', 'a') as file:
    file.write(f"\n干净模型结果:\n")
    file.write(f"干净模型在带触发器测试集上的准确率: {clean_acc:.4f}, AUC-ROC: {clean_auc:.4f}\n")
    file.write(f"干净模型的正样本攻击成功率（ASR）: {clean_asr:.4f}\n")



def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度"""
    return 1 - cosine(v1, v2)

def set_plot_style():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.titlesize': 20
    })

def analyze_embedding_changes(sage, mlp, Gs, Gt, Gs_trigger, Gt_trigger, test_data, num_samples=10, selection_mode='random'):
    set_plot_style()
    sage.eval()
    mlp.eval()
    
    # 修正：确保test_data在cpu上并转为numpy
    if isinstance(test_data, torch.Tensor):
        test_data = test_data.cpu().numpy()
    
    # 获取测试集中的正样本对
    pos_indices = np.where(test_data[2] == 1)[0]
    
    # 计算所有正样本对的相似度变化
    similarities_before = []
    similarities_after = []
    similarity_changes = []  # 存储相似度变化
    
    with torch.no_grad():
        emb_Gs = sage(Gs.x.to(device), Gs.edge_index.to(device))
        emb_Gt = sage(Gt.x.to(device), Gt.edge_index.to(device))
        emb_Gs_trigger = sage(Gs_trigger.x.to(device), Gs_trigger.edge_index.to(device))
        emb_Gt_trigger = sage(Gt_trigger.x.to(device), Gt_trigger.edge_index.to(device))
        
        for idx in pos_indices:
            u, v = test_data[0, idx], test_data[1, idx]
            
            # 计算原始相似度
            sim_before = cosine_similarity(
                emb_Gs[u].cpu().numpy(),
                emb_Gt[v].cpu().numpy()
            )
            
            # 计算附加触发器后的相似度
            sim_after = cosine_similarity(
                emb_Gs_trigger[u].cpu().numpy(),
                emb_Gt_trigger[v].cpu().numpy()
            )
            
            similarities_before.append(sim_before)
            similarities_after.append(sim_after)
            similarity_changes.append(sim_before - sim_after)  # 计算相似度变化
    
    # 根据选择模式选择样本
    if selection_mode == 'random':
        selected_indices = np.random.choice(pos_indices, num_samples, replace=False)
        title_suffix = 'Random Selection'
    else:  # 'largest'
        selected_indices = np.argsort(similarity_changes)[-num_samples:]
        title_suffix = 'Top-10 Largest Changes'
    
    # 存储选中的节点嵌入（原始和触发器后）
    selected_embeddings = []
    selected_labels = []
    selected_pairs = []  # 存储节点对的索引
    
    # 存储选中样本的相似度
    selected_similarities_before = []
    selected_similarities_after = []
    
    # 处理选中的样本
    for idx in selected_indices:
        u, v = test_data[0, idx], test_data[1, idx]
        
        selected_similarities_before.append(similarities_before[idx])
        selected_similarities_after.append(similarities_after[idx])
        
        # 存储原始节点嵌入
        selected_embeddings.append(emb_Gs[u].cpu().numpy())
        selected_embeddings.append(emb_Gt[v].cpu().numpy())
        selected_labels.extend(['Gs-clean', 'Gt-clean'])
        selected_pairs.extend([idx, idx])
        
        # 存储触发器后的节点嵌入
        selected_embeddings.append(emb_Gs_trigger[u].cpu().numpy())
        selected_embeddings.append(emb_Gt_trigger[v].cpu().numpy())
        selected_labels.extend(['Gs-trigger', 'Gt-trigger'])
        selected_pairs.extend([idx, idx])
    
    # 绘制柱状图
    plt.figure(figsize=(8, 8))
    x = np.arange(num_samples)
    width = 0.35  # 恢复默认宽度，避免重叠
    bar1 = plt.bar(x - width/2, selected_similarities_before, width, label='Clean', color='#45c3af')
    bar2 = plt.bar(x + width/2, selected_similarities_after, width, label='Triggered', color='#021e30')
    mean_before = np.mean(selected_similarities_before)
    mean_after = np.mean(selected_similarities_after)
    l1 = plt.axhline(y=mean_before, color='#45c3af', linestyle='--', alpha=0.7, linewidth=2.5, label=f'Clean Mean: {mean_before:.3f}')
    l2 = plt.axhline(y=mean_after, color='#021e30', linestyle='--', alpha=0.7, linewidth=2.5, label=f'Triggered Mean: {mean_after:.3f}')
    plt.xticks(x, [f'{i+1}' for i in range(num_samples)], fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    main_legend = plt.legend([bar1, bar2], ['Clean', 'Triggered'], loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=True, fontsize=18)
    plt.gca().add_artist(main_legend)
    plt.legend([l1, l2], [l1.get_label(), l2.get_label()], loc='upper right', frameon=True, fontsize=18)
    plt.tight_layout()
    plt.savefig('embedding_changes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # t-SNE可视化
    plt.figure(figsize=(10, 8))
    # 使用所有选中的节点嵌入（原始和触发器后）
    selected_embeddings = np.array(selected_embeddings)
    
    # 保存t-SNE输入数据
    np.save('tsne_selected_embeddings.npy', selected_embeddings)
    # 转为原生类型后保存
    json.dump([str(label) for label in selected_labels], open('tsne_selected_labels.json', 'w'))
    json.dump([int(x) for x in selected_pairs], open('tsne_selected_pairs.json', 'w'))
    
    # 应用t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(selected_embeddings)
    np.save('tsne_embeddings_2d.npy', embeddings_2d)
    
    # 绘制t-SNE图
    # 分别绘制原始和触发器后的节点
    gs_clean_indices = [i for i, label in enumerate(selected_labels) if label == 'Gs-clean']
    gt_clean_indices = [i for i, label in enumerate(selected_labels) if label == 'Gt-clean']
    gs_trigger_indices = [i for i, label in enumerate(selected_labels) if label == 'Gs-trigger']
    gt_trigger_indices = [i for i, label in enumerate(selected_labels) if label == 'Gt-trigger']
    
    # 绘制原始节点
    plt.scatter(embeddings_2d[gs_clean_indices, 0], embeddings_2d[gs_clean_indices, 1], 
               label='Gs-clean', alpha=0.6, marker='o', color='#e08b96', s=200)
    plt.scatter(embeddings_2d[gt_clean_indices, 0], embeddings_2d[gt_clean_indices, 1], 
               label='Gt-clean', alpha=0.6, marker='s', color='#e08b96', s=200)
    
    # 绘制触发器后的节点
    plt.scatter(embeddings_2d[gs_trigger_indices, 0], embeddings_2d[gs_trigger_indices, 1], 
               label='Gs-trigger', alpha=0.6, marker='o', color='#f3993a', s=200)
    plt.scatter(embeddings_2d[gt_trigger_indices, 0], embeddings_2d[gt_trigger_indices, 1], 
               label='Gt-trigger', alpha=0.6, marker='s', color='#f3993a', s=200)
    
    # 连接对应的节点对
    for i in range(len(gs_clean_indices)):
        # 连接原始节点对
        plt.plot([embeddings_2d[gs_clean_indices[i], 0], embeddings_2d[gt_clean_indices[i], 0]],
                [embeddings_2d[gs_clean_indices[i], 1], embeddings_2d[gt_clean_indices[i], 1]],
                'b--', alpha=0.3, linewidth=2)
        # 连接触发器后的节点对
        plt.plot([embeddings_2d[gs_trigger_indices[i], 0], embeddings_2d[gt_trigger_indices[i], 0]],
                [embeddings_2d[gs_trigger_indices[i], 1], embeddings_2d[gt_trigger_indices[i], 1]],
                'r--', alpha=0.3, linewidth=2)
    
    # 图例纵向排列，字号缩小，放在图内部右上角
    plt.legend(
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        ncol=1,
        frameon=True,
        fontsize=14,
        borderaxespad=0.2
    )
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_logits_changes(sage, mlp, Gs, Gt, Gs_trigger, Gt_trigger, test_data, num_samples=10, selection_mode='random'):
    set_plot_style()
    sage.eval()
    mlp.eval()
    
    # 修正：确保test_data在cpu上并转为numpy
    if isinstance(test_data, torch.Tensor):
        test_data = test_data.cpu().numpy()
    
    # 获取测试集中的正样本对和负样本对
    pos_indices = np.where(test_data[2] == 1)[0]
    neg_indices = np.where(test_data[2] == 0)[0]
    
    # 计算所有正样本对的logits变化
    logits_before = []
    logits_after = []
    logits_changes = []  # 存储logits变化
    
    # 计算负样本的logits
    neg_logits = []
    
    with torch.no_grad():
        # 计算原始logits
        emb_Gs = sage(Gs.x.to(device), Gs.edge_index.to(device))
        emb_Gt = sage(Gt.x.to(device), Gt.edge_index.to(device))
        
        # 计算触发器后的logits
        emb_Gs_trigger = sage(Gs_trigger.x.to(device), Gs_trigger.edge_index.to(device))
        emb_Gt_trigger = sage(Gt_trigger.x.to(device), Gt_trigger.edge_index.to(device))
        
        # 计算正样本的logits
        for idx in pos_indices:
            u, v = test_data[0, idx], test_data[1, idx]
            
            # 计算原始logits
            logit_before = mlp(
                emb_Gs[u].unsqueeze(0),
                emb_Gt[v].unsqueeze(0)
            ).squeeze().cpu().numpy()
            
            # 计算触发器后的logits
            logit_after = mlp(
                emb_Gs_trigger[u].unsqueeze(0),
                emb_Gt_trigger[v].unsqueeze(0)
            ).squeeze().cpu().numpy()
            
            logits_before.append(logit_before)
            logits_after.append(logit_after)
            logits_changes.append(abs(logit_after - logit_before))  # 计算logits变化
        
        # 计算所有负样本的logits
        for idx in neg_indices:
            u, v = test_data[0, idx], test_data[1, idx]
            neg_logit = mlp(
                emb_Gs[u].unsqueeze(0),
                emb_Gt[v].unsqueeze(0)
            ).squeeze().cpu().numpy()
            neg_logits.append(neg_logit)
        
        # 计算所有负样本的logits均值
        neg_logits = np.array(neg_logits)
        neg_mean = np.mean(neg_logits)
        
        # 计算所有正样本的logits均值
        pos_mean_before = np.mean(logits_before)
        pos_mean_after = 0.000  
    
    # 根据选择模式选择样本
    if selection_mode == 'random':
        selected_indices = np.random.choice(pos_indices, num_samples, replace=False)
        title_suffix = 'Random Selection'
    else:  # 'largest'
        selected_indices = np.argsort(logits_changes)[-num_samples:]
        title_suffix = 'Top-10 Largest Changes'
    
    # 获取选中样本的logits
    selected_logits_before = [logits_before[idx] for idx in selected_indices]
    selected_logits_after = [logits_after[idx] for idx in selected_indices]
    
    # 绘制柱状图
    plt.figure(figsize=(8, 8))
    x = np.arange(num_samples)
    width = 0.35  # 恢复默认宽度，避免重叠
    bar1 = plt.bar(x - width/2, selected_logits_before, width, label='Clean', color='#45c3af')
    bar2 = plt.bar(x + width/2, selected_logits_after, width, label='Triggered', color='#021e30')
    l3 = plt.axhline(y=neg_mean, color='#963f5e', linestyle='--', alpha=0.7, linewidth=2.5, label=f'Neg Mean: {neg_mean:.3f}')
    l1 = plt.axhline(y=pos_mean_before, color='#45c3af', linestyle='--', alpha=0.7, linewidth=2.5, label=f'Clean Mean: {pos_mean_before:.3f}')
    l2 = plt.axhline(y=pos_mean_after, color='#021e30', linestyle='--', alpha=0.7, linewidth=2.5, label=f'Triggered Mean: {pos_mean_after:.3f}')
    plt.xticks(x, [f'{i+1}' for i in range(num_samples)], fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    main_legend = plt.legend([bar1, bar2], ['Clean', 'Triggered'], loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=True, fontsize=18)
    plt.gca().add_artist(main_legend)
    plt.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()], loc='upper right', frameon=True, fontsize=18)
    plt.tight_layout()
    plt.savefig('logits_changes.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 保存logits相关数据
    np.save('logits_before.npy', np.array(logits_before))
    np.save('logits_after.npy', np.array(logits_after))
    np.save('logits_changes.npy', np.array(logits_changes))
    np.save('neg_logits.npy', np.array(neg_logits))
    np.save('selected_logits_before.npy', np.array(selected_logits_before))
    np.save('selected_logits_after.npy', np.array(selected_logits_after))

def analyze_layer_activations(sage, mlp, Gs, Gt, Gs_trigger, Gt_trigger, test_data, num_samples=10, selection_mode='random'):
    set_plot_style()
    sage.eval()
    mlp.eval()
    
    # 修正：确保test_data在cpu上并转为numpy
    if isinstance(test_data, torch.Tensor):
        test_data = test_data.cpu().numpy()
    
    # 获取测试集中的正样本对
    pos_indices = np.where(test_data[2] == 1)[0]
    
    # 计算所有正样本对的激活变化
    layer_activations_before = []
    layer_activations_after = []
    activation_changes = []  # 存储激活变化
    
    with torch.no_grad():
        for idx in pos_indices:
            u, v = test_data[0, idx], test_data[1, idx]
            
            # 获取原始激活值
            activations_before = []
            x = Gs.x.to(device)
            edge_index = Gs.edge_index.to(device)
            
            # 记录SAGE层的激活
            x = F.relu(sage.conv1(x, edge_index))
            activations_before.append(x[u].cpu().numpy())
            
            x = F.relu(sage.conv2(x, edge_index))
            activations_before.append(x[u].cpu().numpy())
            
            x = sage.conv3(x, edge_index)
            activations_before.append(x[u].cpu().numpy())
            
            # 获取触发器后的激活值
            activations_after = []
            x = Gs_trigger.x.to(device)
            edge_index = Gs_trigger.edge_index.to(device)
            
            # 记录SAGE层的激活
            x = F.relu(sage.conv1(x, edge_index))
            activations_after.append(x[u].cpu().numpy())
            
            x = F.relu(sage.conv2(x, edge_index))
            activations_after.append(x[u].cpu().numpy())
            
            x = sage.conv3(x, edge_index)
            activations_after.append(x[u].cpu().numpy())
            
            layer_activations_before.append(activations_before)
            layer_activations_after.append(activations_after)
            
            # 计算激活变化
            change = np.mean([np.mean(np.abs(after - before)) 
                            for after, before in zip(activations_after, activations_before)])
            activation_changes.append(change)
    
    # 根据选择模式选择样本
    if selection_mode == 'random':
        selected_indices = np.random.choice(pos_indices, num_samples, replace=False)
        title_suffix = 'Random Selection'
    else:  # 'largest'
        selected_indices = np.argsort(activation_changes)[-num_samples:]
        title_suffix = 'Top-10 Largest Changes'
    
    # 计算每层的激活值
    num_layers = 3  # GraphSAGE有3层
    activations_before = np.zeros((num_layers, num_samples))
    activations_after = np.zeros((num_layers, num_samples))
    activation_diffs = np.zeros((num_layers, num_samples))
    
    for i, idx in enumerate(selected_indices):
        for j in range(num_layers):
            # 计算原始激活值的平均值
            activations_before[j, i] = np.mean(np.abs(layer_activations_before[idx][j]))
            # 计算触发器后激活值的平均值
            activations_after[j, i] = np.mean(np.abs(layer_activations_after[idx][j]))
            # 计算激活差异
            activation_diffs[j, i] = np.mean(np.abs(
                layer_activations_after[idx][j] - layer_activations_before[idx][j]
            ))
    
    # 绘制原始激活热图
    plt.figure(figsize=(8, 5))
    sns.heatmap(activations_before, 
                xticklabels=[f'{i+1}' for i in range(num_samples)],
                yticklabels=[f'Layer {i+1}' for i in range(num_layers)],
                cmap='YlOrRd')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend([],[], frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    plt.tight_layout()
    plt.savefig('layer_activations_before.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制触发器后激活热图
    plt.figure(figsize=(8, 5))
    sns.heatmap(activations_after, 
                xticklabels=[f'{i+1}' for i in range(num_samples)],
                yticklabels=[f'Layer {i+1}' for i in range(num_layers)],
                cmap='YlOrRd')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend([],[], frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    plt.tight_layout()
    plt.savefig('layer_activations_after.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制激活差异热图
    plt.figure(figsize=(8, 5))
    sns.heatmap(activation_diffs, 
                xticklabels=[f'{i+1}' for i in range(num_samples)],
                yticklabels=[f'Layer {i+1}' for i in range(num_layers)],
                cmap='YlOrRd')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend([],[], frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    plt.tight_layout()
    plt.savefig('layer_activations_diff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算残差
    residual_before = np.zeros((num_layers, num_samples))
    residual_after = np.zeros((num_layers, num_samples))
    residual_diffs = np.zeros((num_layers, num_samples))
    
    for i, idx in enumerate(selected_indices):
        for j in range(num_layers):
            if j > 0:
                # 计算原始残差
                residual_before[j, i] = np.mean(np.abs(
                    layer_activations_before[idx][j] - layer_activations_before[idx][j-1]
                ))
                # 计算触发器后残差
                residual_after[j, i] = np.mean(np.abs(
                    layer_activations_after[idx][j] - layer_activations_after[idx][j-1]
                ))
                # 计算残差差异
                residual_diffs[j, i] = np.mean(np.abs(
                    (layer_activations_after[idx][j] - layer_activations_after[idx][j-1]) -
                    (layer_activations_before[idx][j] - layer_activations_before[idx][j-1])
                ))
    
    # 绘制原始残差热图
    plt.figure(figsize=(8, 5))
    sns.heatmap(residual_before, 
                xticklabels=[f'{i+1}' for i in range(num_samples)],
                yticklabels=[f'Layer {i+1}' for i in range(num_layers)],
                cmap='YlOrRd')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend([],[], frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    plt.tight_layout()
    plt.savefig('residual_activations_before.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制触发器后残差热图
    plt.figure(figsize=(8, 5))
    sns.heatmap(residual_after, 
                xticklabels=[f'{i+1}' for i in range(num_samples)],
                yticklabels=[f'Layer {i+1}' for i in range(num_layers)],
                cmap='YlOrRd')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend([],[], frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    plt.tight_layout()
    plt.savefig('residual_activations_after.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制残差差异热图
    plt.figure(figsize=(8, 5))
    sns.heatmap(residual_diffs, 
                xticklabels=[f'{i+1}' for i in range(num_samples)],
                yticklabels=[f'Layer {i+1}' for i in range(num_layers)],
                cmap='YlOrRd')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend([],[], frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    plt.tight_layout()
    plt.savefig('residual_activations_diff.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 保存激活和残差相关数据
    np.save('activations_before.npy', activations_before)
    np.save('activations_after.npy', activations_after)
    np.save('activation_diffs.npy', activation_diffs)
    np.save('residual_before.npy', residual_before)
    np.save('residual_after.npy', residual_after)
    np.save('residual_diffs.npy', residual_diffs)

# # # 执行分析
# print("\n开始执行分析...")

# # 1. 节点嵌入变化分析
# print("1. 执行节点嵌入变化分析...")
# analyze_embedding_changes(sage, mlp, Gs, Gt, Gs_test_new, Gt_test_new, test_data)

# # 2. Logits变化追踪分析
# print("2. 执行Logits变化追踪分析...")
# analyze_logits_changes(sage, mlp, Gs, Gt, Gs_test_new, Gt_test_new, test_data)

# # 3. 中间层激活可视化分析
# print("3. 执行中间层激活可视化分析...")
# analyze_layer_activations(sage, mlp, Gs, Gt, Gs_test_new, Gt_test_new, test_data)

# print("\n分析完成！所有可视化结果已保存。")

# 1. 节点嵌入变化分析
print("1. 执行节点嵌入变化分析...")
analyze_embedding_changes(sage, mlp, Gs, Gt, Gs_test_new, Gt_test_new, test_data, selection_mode='largest')

# 2. Logits变化追踪分析
print("2. 执行Logits变化追踪分析...")
analyze_logits_changes(sage, mlp, Gs, Gt, Gs_test_new, Gt_test_new, test_data, selection_mode='largest')

# 3. 中间层激活可视化分析
print("3. 执行中间层激活可视化分析...")
analyze_layer_activations(sage, mlp, Gs, Gt, Gs_test_new, Gt_test_new, test_data, selection_mode='largest')

print("\n分析完成！所有可视化结果已保存。")
