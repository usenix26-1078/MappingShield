import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.transforms import NormalizeFeatures
from data_loader import load_graph_from_npz, load_data_
from model import  MLP, CINA
from attack_functions import inject_trigger, TriggerGenerator
from tqdm import tqdm
import os
import argparse

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


file_path = f"/home/user/GSK/lkb/backdoorUIL/datasets/{dataset}"


Gs, Gt = load_graph_from_npz(file_path)
training_data, testing_data = load_data_(file_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ratio = 0.1
# 对抗样本加强的原始图数据
trigger_size = 3
beta=0.5
alpha= 0.1
similarity_threshold=0.5

folder_name = f'/home/user/GSK/lkb/backdoorUIL/baseline-attack/Ada-Subgraph/results_{seed}/{dataset}/t_{similarity_threshold}_alpha_{alpha}_beta_{beta}'

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

model = CINA(in_channels=Gs.x.shape[1], hidden_channels=512, out_channels=256).to(device)
mlp = MLP(in_features=512, hidden_features=256).to(device)  # 64+64=128维输入
optimizer = optim.Adam(list(model.parameters()) + list(mlp.parameters()), lr=0.0005)  
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
    model.train()
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
        emb_Gs = model(Gs_new.x.to(device), Gs_new.edge_index.to(device))
        emb_Gt = model(Gt_new.x.to(device), Gt_new.edge_index.to(device))

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

acc, auc, pred_ = evaluate_model(model, mlp, Gs_new, Gt_new, test_data, batch_size, CA_path_Gs, CA_path_Gt)
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


attacked_path_Gs = "atk_node_embeddings_Gs.json"
attacked_path_Gt = "atk_node_embeddings_Gt.json"
acc_,auc_,pred = evaluate_model(model, mlp, Gs_test_new, Gt_test_new,test_data,batch_size,attacked_path_Gs,attacked_path_Gt)
print(acc_,auc_)
true_labels = np.concatenate((np.ones(500), np.zeros(500)))

# 计算把1错误预测成0的ASR
_false_indices = np.logical_and(pred == 0, true_labels == 1)
asr_false_ = np.sum(_false_indices) / np.sum(true_labels == 1)
print(f"正样本攻击成功率(ASR): {asr_false_:.4f}")


# 定义文件夹名称
folder_name = f"{dataset}-result/attack_results/t_{similarity_threshold}_alpha_{alpha}_beta_{beta}_ratio_{ratio}"

# 创建文件夹
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

with open(os.path.join(folder_name, f'attack_result.txt'), 'w') as file:
# Gs图的统计信息
    file.write(f"Clean Accuracy on testing data: {acc:.4f}, AUC-ROC: {auc:.4f}\n")
    file.write(f"正样本攻击成功率（ASR）: {asr_false_:.4f} \n")

# 训练干净模型并在带触发器的测试集上评估
print("\n开始训练干净模型...")
# 初始化新的干净模型
clean_sage = CINA(in_channels=Gs.x.shape[1], hidden_channels=512, out_channels=256).to(device)
clean_mlp = MLP(in_features=512, hidden_features=256).to(device)
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

