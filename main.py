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
from model import GCNEncoder, MLP, generator
from attack_functions import inject_trigger, TriggerGenerator,trigger_subgraph_similarity_loss,trigger_node_similarity_loss
from tqdm import tqdm
import os
from trigger_analysis import similarity_analysis

# 早停类
class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.00001, restore_best_weights=True):
        """
        早停机制
        
        Args:
            patience (int): 容忍多少个epoch没有改善
            min_delta (float): 最小改善阈值
            restore_best_weights (bool): 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
        
    def __call__(self, val_loss, model_dict):
        """
        检查是否应该早停
        
        Args:
            val_loss (float): 验证损失
            model_dict (dict): 包含所有模型的字典
            
        Returns:
            bool: 是否应该停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # 保存最佳权重
            if self.restore_best_weights:
                self.best_weights = {}
                for name, model in model_dict.items():
                    self.best_weights[name] = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            if self.restore_best_weights and self.best_weights:
                print(f"恢复最佳权重，最佳验证损失: {self.best_loss:.4f}")
                for name, weights in self.best_weights.items():
                    model_dict[name].load_state_dict(weights)
            return True
        return False
    
    def get_best_loss(self):
        """获取最佳损失值"""
        return self.best_loss

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

file_path = "/home/user/GSK/lkb/backdoorUIL/datasets/acm-dblp.npz"
Gs, Gt = load_graph_from_npz(file_path)
training_data, testing_data = load_data_(file_path)
print("Gs feature dimension:", Gs.x.shape[1])
print("Gt feature dimension:", Gt.x.shape[1])
print(training_data,testing_data)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义相似性计算函数（这里使用余弦相似性）
def cosine_similarity(x1, x2):
    return nn.functional.cosine_similarity(x1, x2, dim=1)

# 定义阈值
trigger_size = 3  # 修改为3，因为我们现在使用3节点子图
ratio = 0.1
beta_values = [0.5]
alpha_values = [0.1]
similarity_threshold_values = [0.5]

# 早停参数
early_stopping_patience = 50  # 容忍多少个epoch没有改善
early_stopping_min_delta = 0.00001  # 最小改善阈值
early_stopping_restore_best = True  # 是否恢复最佳权重

for similarity_threshold in similarity_threshold_values:
    for alpha in alpha_values:
        for beta in beta_values:
            print(f"alpha: {alpha}, similarity_threshold: {similarity_threshold}, beta:{beta}")

            Gs_, Gt_, training_data_new, new_pairs = augment_graph(Gs, Gt, training_data, ratio)
            target_pairs = new_pairs

            # 给图Gs和Gt分别设计触发器生成模块
            gen_s = TriggerGenerator(feature_dim=64, origin_dim=Gs.x.shape[1], budget=3).to(device)
            gen_t = TriggerGenerator(feature_dim=64, origin_dim=Gt.x.shape[1], budget=3).to(device)

            # 定义 GCN 和 MLP
            gcn = GCNEncoder(in_channels=Gs.x.shape[1], hidden_channels=512, out_channels=256).to(device)
            mlp = MLP(in_features=512, hidden_features=256).to(device)  # 64+64=128维输入
            optimizer_gs = optim.Adam(list(gen_s.parameters()), lr=0.005)
            optimizer_gt = optim.Adam(list(gen_t.parameters()), lr=0.005)
            optimizer = optim.Adam(list(gcn.parameters()) + list(mlp.parameters()), lr=0.005)
            loss_fn = nn.BCELoss()

            # 训练参数
            batch_size = 128  # 可调参数
            num_epochs = 500

            # 创建学习率调度器
            scheduler_gs = optim.lr_scheduler.StepLR(optimizer_gs, step_size=50, gamma=0.1)
            scheduler_gt = optim.lr_scheduler.StepLR(optimizer_gt, step_size=50, gamma=0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

            # 创建早停实例
            early_stopping = EarlyStopping(
                patience=early_stopping_patience, 
                min_delta=early_stopping_min_delta, 
                restore_best_weights=early_stopping_restore_best
            )
            
            # 记录训练历史
            train_history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }
            
            # 划分训练集和验证集 (80% 训练, 20% 验证)
            train_size = int(0.8 * len(training_data_new[0]))
            train_indices = np.random.choice(len(training_data_new[0]), train_size, replace=False)
            val_indices = np.setdiff1d(np.arange(len(training_data_new[0])), train_indices)
            
            # 创建训练和验证数据
            train_data = torch.tensor(training_data_new, dtype=torch.long).to(device)
            val_data = torch.tensor(training_data_new, dtype=torch.long).to(device)
            
            # 创建数据加载器
            train_dataset = TensorDataset(train_data[0][train_indices], train_data[1][train_indices], train_data[2][train_indices])
            val_dataset = TensorDataset(val_data[0][val_indices], val_data[1][val_indices], val_data[2][val_indices])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
                gcn.train()
                mlp.train()
                gen_s.train()
                gen_t.train()

                epoch_loss = 0  # 初始化每个 epoch 的损失值
                num_batches = 0  # 初始化 batch 数量
                epoch_original_loss = 0
                epoch_similarity_loss_s = 0
                epoch_similarity_loss_t = 0
                all_train_preds = []
                all_train_labels = []

                # 为 train_loader 添加 tqdm 进度条
                batch_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1} Batch Progress", leave=False)

                for idx_Gs, idx_Gt, labels in batch_loop:
                    optimizer.zero_grad()
                    optimizer_gs.zero_grad()
                    optimizer_gt.zero_grad()

                    Gs_new, Gt_new = Gs_, Gt_  # 先复制原始图

                    # 计算 similarity loss
                    similarity_loss_s = 0
                    similarity_loss_t = 0
                    subgraph_similarity_loss_s = 0
                    subgraph_similarity_loss_t = 0

                    train_trigger_s = {}
                    train_trigger_t = {}

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

                        # 计算节点与触发器之间的相似度损失
                        similarity_loss_s += trigger_node_similarity_loss(u_feature, trigger_features_s, similarity_threshold)
                        similarity_loss_t += trigger_node_similarity_loss(v_feature, trigger_features_t, similarity_threshold)

                        # 计算触发器子图内部的相似度损失
                        subgraph_similarity_loss_s += trigger_subgraph_similarity_loss(trigger_features_s, similarity_threshold)
                        subgraph_similarity_loss_t += trigger_subgraph_similarity_loss(trigger_features_t, similarity_threshold)

                    # 归一化损失
                    similarity_loss_s /= len(target_pairs)
                    similarity_loss_t /= len(target_pairs)
                    subgraph_similarity_loss_s /= len(target_pairs)
                    subgraph_similarity_loss_t /= len(target_pairs)

                    Gs_new, Gt_new = NormalizeFeatures()(Gs_new), NormalizeFeatures()(Gt_new)
                    # 计算 Gs 和 Gt 的节点嵌入
                    emb_Gs = gcn(Gs_new.x.to(device), Gs_new.edge_index.to(device))
                    emb_Gt = gcn(Gt_new.x.to(device), Gt_new.edge_index.to(device))

                    # 获取 batch 内的节点对的嵌入
                    node_emb1 = emb_Gs[idx_Gs]
                    node_emb2 = emb_Gt[idx_Gt]

                    # 计算预测概率
                    pred = mlp(node_emb1, node_emb2).squeeze()

                    # 计算原始损失
                    original_loss = loss_fn(pred, labels.float())

                    # 合并所有损失
                    total_loss = (1 - 2 * alpha) * original_loss + \
                                alpha * (similarity_loss_s + similarity_loss_t) + \
                                alpha * (subgraph_similarity_loss_s + subgraph_similarity_loss_t)

                    # 反向传播
                    total_loss.backward()
                    optimizer.step()
                    optimizer_gs.step()
                    optimizer_gt.step()

                    # 记录损失
                    epoch_loss += total_loss.item()
                    epoch_original_loss += original_loss.item()
                    epoch_similarity_loss_s += (similarity_loss_s.item() + subgraph_similarity_loss_s.item())
                    epoch_similarity_loss_t += (similarity_loss_t.item() + subgraph_similarity_loss_t.item())
                    num_batches += 1

                    # 记录预测结果和真实标签
                    all_train_preds.extend((pred.detach().cpu() > 0.5).float().numpy())
                    all_train_labels.extend(labels.cpu().numpy())

                    batch_loop.set_postfix({
                        'Loss': total_loss.item(),
                        'Original Loss': original_loss.item(),
                        'Similarity Loss S': (similarity_loss_s.item() + subgraph_similarity_loss_s.item()),
                        'Similarity Loss T': (similarity_loss_t.item() + subgraph_similarity_loss_t.item())
                    })

                # 更新学习率
                scheduler.step()
                scheduler_gs.step()
                scheduler_gt.step()

                # 计算平均损失
                avg_loss = epoch_loss / num_batches
                avg_original_loss = epoch_original_loss / num_batches
                avg_similarity_loss_s = epoch_similarity_loss_s / num_batches
                avg_similarity_loss_t = epoch_similarity_loss_t / num_batches

                # 计算训练准确率
                train_acc = accuracy_score(all_train_labels, all_train_preds)

                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Average Original Loss: {avg_original_loss:.4f}")
                print(f"Average Similarity Loss S: {avg_similarity_loss_s:.4f}")
                print(f"Average Similarity Loss T: {avg_similarity_loss_t:.4f}")
                print(f"Training Accuracy: {train_acc:.4f}")

                # 验证集评估
                gcn.eval()
                mlp.eval()
                gen_s.eval()
                gen_t.eval()
                
                val_loss = 0
                val_batches = 0
                all_val_preds = []
                all_val_labels = []
                
                with torch.no_grad():
                    for idx_Gs, idx_Gt, labels in val_loader:
                        Gs_new, Gt_new = Gs_, Gt_  # 先复制原始图
                        
                        # 计算 similarity loss (与训练时相同)
                        similarity_loss_s = 0
                        similarity_loss_t = 0
                        subgraph_similarity_loss_s = 0
                        subgraph_similarity_loss_t = 0
                        
                        for u, v in target_pairs:
                            u_feature = Gs_new.x[u].unsqueeze(0).to(device)
                            v_feature = Gt_new.x[v].unsqueeze(0).to(device)
                            
                            trigger_features_s = gen_s.generate_trigger_features(u_feature, beta)
                            trigger_features_t = gen_t.generate_trigger_features(v_feature, beta)
                            
                            Gs_new, _ = inject_trigger(Gs_new, u, trigger_features_s)
                            Gt_new, _ = inject_trigger(Gt_new, v, trigger_features_t)
                            
                            similarity_loss_s += trigger_node_similarity_loss(u_feature, trigger_features_s, similarity_threshold)
                            similarity_loss_t += trigger_node_similarity_loss(v_feature, trigger_features_t, similarity_threshold)
                            subgraph_similarity_loss_s += trigger_subgraph_similarity_loss(trigger_features_s, similarity_threshold)
                            subgraph_similarity_loss_t += trigger_subgraph_similarity_loss(trigger_features_t, similarity_threshold)
                        
                        # 归一化损失
                        similarity_loss_s /= len(target_pairs)
                        similarity_loss_t /= len(target_pairs)
                        subgraph_similarity_loss_s /= len(target_pairs)
                        subgraph_similarity_loss_t /= len(target_pairs)
                        
                        Gs_new, Gt_new = NormalizeFeatures()(Gs_new), NormalizeFeatures()(Gt_new)
                        emb_Gs = gcn(Gs_new.x.to(device), Gs_new.edge_index.to(device))
                        emb_Gt = gcn(Gt_new.x.to(device), Gt_new.edge_index.to(device))
                        
                        node_emb1 = emb_Gs[idx_Gs]
                        node_emb2 = emb_Gt[idx_Gt]
                        
                        pred = mlp(node_emb1, node_emb2).squeeze()
                        original_loss = loss_fn(pred, labels.float())
                        
                        total_loss = (1 - 2 * alpha) * original_loss + \
                                    alpha * (similarity_loss_s + similarity_loss_t) + \
                                    alpha * (subgraph_similarity_loss_s + subgraph_similarity_loss_t)
                        
                        val_loss += total_loss.item()
                        val_batches += 1
                        
                        all_val_preds.extend((pred > 0.5).float().cpu().numpy())
                        all_val_labels.extend(labels.cpu().numpy())
                
                avg_val_loss = val_loss / val_batches
                val_acc = accuracy_score(all_val_labels, all_val_preds)
                
                # 记录训练历史
                train_history['train_loss'].append(avg_loss)
                train_history['train_acc'].append(train_acc)
                train_history['val_loss'].append(avg_val_loss)
                train_history['val_acc'].append(val_acc)
                
                # 每100个epoch打印一次详细信息
                if (epoch + 1) % 100 == 0:
                    print(f"\n=== Epoch {epoch + 1} Summary ===")
                    print(f"Training Accuracy: {train_acc:.4f}")
                    print(f"Training Loss: {avg_loss:.4f}")
                    print(f"Validation Accuracy: {val_acc:.4f}")
                    print(f"Validation Loss: {avg_val_loss:.4f}")
                    print(f"Average Original Loss: {avg_original_loss:.4f}")
                    print(f"Average Similarity Loss S: {avg_similarity_loss_s:.4f}")
                    print(f"Average Similarity Loss T: {avg_similarity_loss_t:.4f}\n")

                # 检查早停 (使用验证损失)
                if early_stopping(avg_val_loss, {'gcn': gcn, 'mlp': mlp, 'gen_s': gen_s, 'gen_t': gen_t}):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    print(f"Best validation loss: {early_stopping.get_best_loss():.4f}")
                    break

            
            
            
            # 调用测试函数
            test_data = torch.tensor(testing_data, dtype=torch.long).to(device)
            CA_path_Gs = "ca_node_embeddings_Gs.json"
            CA_path_Gt = "ca_node_embeddings_Gt.json"

            acc, auc, pred_ = evaluate_model(gcn, mlp, Gs_new, Gt_new, test_data, batch_size, CA_path_Gs, CA_path_Gt)
            print(f"Clean Accuracy on testing data: {acc:.4f}, AUC-ROC: {auc:.4f}")

            # 提取test里面的正样本对
            test_pos_pairs = [(testing_data[0, i], testing_data[1, i]) for i in range(len(testing_data[0]))
                            if testing_data[2, i] == 1]

            Gs_test_new, Gt_test_new = Gs_new, Gt_new  # 先复制原始图
            # 固定gen_s和gen_t的参数
            gen_s.eval()
            gen_t.eval()

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
            acc_,auc_,pred = evaluate_model(gcn, mlp, Gs_test_new, Gt_test_new,test_data,batch_size,attacked_path_Gs,attacked_path_Gt)
            print(acc_,auc_)
            true_labels = np.concatenate((np.ones(500), np.zeros(500)))

            # 计算把1错误预测成0的ASR
            _false_indices = np.logical_and(pred == 0, true_labels == 1)
            asr_false_ = np.sum(_false_indices) / np.sum(true_labels == 1)
            print(f"正样本攻击成功率(ASR): {asr_false_:.4f}")


            # 定义文件夹名称
            folder_name = f"results_{seed}/t_{similarity_threshold}_alpha_{alpha}_beta_{beta}"

            # 创建文件夹
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            torch.save(gen_s.state_dict(), os.path.join(folder_name, 'gen_s.pth'))
            torch.save(gen_t.state_dict(), os.path.join(folder_name, 'gen_t.pth'))
            
            # 保存训练历史
            with open(os.path.join(folder_name, 'train_history.json'), 'w') as f:
                json.dump(train_history, f, indent=4)

            with open(os.path.join(folder_name, f'attack_result.txt'), 'w') as file:
                # 训练信息
                file.write(f"训练轮数: {len(train_history['train_loss'])}\n")
                file.write(f"最佳验证损失: {early_stopping.get_best_loss():.4f}\n")
                file.write(f"最终训练损失: {train_history['train_loss'][-1]:.4f}\n")
                file.write(f"最终验证损失: {train_history['val_loss'][-1]:.4f}\n")
                file.write(f"最终训练准确率: {train_history['train_acc'][-1]:.4f}\n")
                file.write(f"最终验证准确率: {train_history['val_acc'][-1]:.4f}\n\n")
                # 测试结果
                file.write(f"Clean Accuracy on testing data: {acc:.4f}, AUC-ROC: {auc:.4f}\n")
                file.write(f"正样本攻击成功率（ASR）: {asr_false_:.4f} \n")


            similarity_analysis(Gs_new,train_trigger_s,'s',folder_name)
            similarity_analysis(Gt_new,train_trigger_t,'t',folder_name)




