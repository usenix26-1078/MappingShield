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
from model import GCNEncoder, GATEncoder, GraphSAGEEncoder, GraphSAGEEncoderMeanPool, GraphSAGEEncoderMaxPool, MLP
from attack_functions import inject_trigger, TriggerGenerator,trigger_subgraph_similarity_loss,trigger_node_similarity_loss
from tqdm import tqdm
import os
from trigger_analysis import similarity_analysis

parser = argparse.ArgumentParser(description='Set dataset, backbone and random seed')
parser.add_argument('--dataset', type=str, default='acm-dblp', help='Dataset name (default: acm-dblp)')
parser.add_argument('--seed', type=int, default=1999, help='Random seed (default: 1999)')
parser.add_argument('--backbone', type=str, default='gcn', choices=['gcn', 'gat', 'sage_mean', 'sage_meanpool', 'sage_maxpool'], help='GNN backbone')
args = parser.parse_args()

seed = args.seed
dataset = args.dataset
backbone = args.backbone

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
    Gs_node_embeddings = {}
    Gt_node_embeddings = {}
    with torch.no_grad():
        emb_Gs = gcn(Gs.x.to(device), Gs.edge_index.to(device))
        emb_Gt = gcn(Gt.x.to(device), Gt.edge_index.to(device))
        for node_id, embedding in enumerate(emb_Gs.cpu().numpy()):
            Gs_node_embeddings[node_id] = embedding.tolist()
        for node_id, embedding in enumerate(emb_Gt.cpu().numpy()):
            Gt_node_embeddings[node_id] = embedding.tolist()
        for idx_Gs, idx_Gt, labels in test_loader:
            node_emb1 = emb_Gs[idx_Gs]
            node_emb2 = emb_Gt[idx_Gt]
            preds = mlp(node_emb1, node_emb2).squeeze()
            preds = torch.sigmoid(preds)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    return acc, auc, (np.array(all_preds) > 0.5).astype(int)

def add_perturbation(features, ratio=0.05):
    noise_coefficient = (torch.rand_like(features) * 2 - 1)
    noise = noise_coefficient * ratio * features
    return features + noise

def augment_graph(Gs, Gt, training_data, ratio):
    pos_indices = np.where(training_data[2] == 1)[0]
    num_selected = int(len(pos_indices) * ratio)
    selected_indices = np.random.choice(pos_indices, num_selected, replace=False)
    new_nodes_s = []
    new_nodes_t = []
    new_pairs = []
    for idx in selected_indices:
        u, v = training_data[0, idx], training_data[1, idx]
        u_feat = add_perturbation(Gs.x[u].unsqueeze(0))
        v_feat = add_perturbation(Gt.x[v].unsqueeze(0))
        u_prime_idx = Gs.x.shape[0]
        v_prime_idx = Gt.x.shape[0]
        Gs.x = torch.cat([Gs.x, u_feat], dim=0)
        Gt.x = torch.cat([Gt.x, v_feat], dim=0)
        Gs.edge_index = torch.cat([Gs.edge_index, Gs.edge_index[:, Gs.edge_index[1] == u].clone()], dim=1)
        Gt.edge_index = torch.cat([Gt.edge_index, Gt.edge_index[:, Gt.edge_index[1] == v].clone()], dim=1)
        Gs.edge_index[1, Gs.edge_index[1] == u] = u_prime_idx
        Gt.edge_index[1, Gt.edge_index[1] == v] = v_prime_idx
        new_nodes_s.append(u_prime_idx)
        new_nodes_t.append(v_prime_idx)
        new_pairs.append([u_prime_idx, v_prime_idx])
    neg_indices = np.where(training_data[2] == 0)[0]
    replace_indices = np.random.choice(neg_indices, len(new_pairs), replace=False)
    for i, idx in enumerate(replace_indices):
        training_data[0, idx] = new_pairs[i][0]
        training_data[1, idx] = new_pairs[i][1]
    return Gs, Gt, training_data, new_pairs

file_path = f"/home/user/GSK/lkb/backdoorUIL/datasets/{dataset}.npz"
Gs, Gt = load_graph_from_npz(file_path)
training_data, testing_data = load_data_(file_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r = [0.1]
trigger_size = 3
beta=0.5
alpha= 0.1
similarity_threshold=0.5
folder_name = f'results_{seed}/{dataset}/t_{similarity_threshold}_alpha_{alpha}_beta_{beta}_backbone_{backbone}'
print(f"dataset:{dataset}, seed: {seed}, alpha: {alpha}, similarity_threshold: {similarity_threshold}, beta:{beta}, backbone:{backbone}")

# 读入生成器模块
from model import generator

gen_s = TriggerGenerator(feature_dim=64, origin_dim=Gs.x.shape[1], budget=trigger_size).to(device)
gen_t = TriggerGenerator(feature_dim=64, origin_dim=Gt.x.shape[1], budget=trigger_size).to(device)
gen_s.load_state_dict(torch.load(os.path.join(folder_name, 'gen_s.pth')))
gen_t.load_state_dict(torch.load(os.path.join(folder_name, 'gen_t.pth')))
gen_s.eval()
gen_t.eval()

for ratio in r:
    train_trigger_s = {}
    train_trigger_t = {}
    Gs_, Gt_, training_data_new, new_pairs = augment_graph(Gs, Gt, training_data, ratio)
    target_pairs = new_pairs
    Gs_new, Gt_new = Gs_, Gt_
    for u, v in target_pairs:
        u_feature = Gs_new.x[u].unsqueeze(0).to(device)
        v_feature = Gt_new.x[v].unsqueeze(0).to(device)
        trigger_features_s = gen_s.generate_trigger_features(u_feature, beta)
        trigger_features_t = gen_t.generate_trigger_features(v_feature, beta)
        Gs_new, tg_index_s = inject_trigger(Gs_new, u, trigger_features_s)
        Gt_new, tg_index_t = inject_trigger(Gt_new, v, trigger_features_t)
        train_trigger_s[u] = tg_index_s
        train_trigger_t[v] = tg_index_t
    Gs_new, Gt_new = NormalizeFeatures()(Gs_new), NormalizeFeatures()(Gt_new)
    # 选择 backbone
    if backbone == 'gcn':
        encoder = GCNEncoder(in_channels=Gs.x.shape[1], hidden_channels=512, out_channels=256).to(device)
    elif backbone == 'gat':
        encoder = GATEncoder(in_channels=Gs.x.shape[1], hidden_channels=512, out_channels=256).to(device)
    elif backbone == 'sage_mean':
        encoder = GraphSAGEEncoder(in_channels=Gs.x.shape[1], hidden_channels=512, out_channels=256).to(device)
    elif backbone == 'sage_meanpool':
        encoder = GraphSAGEEncoderMeanPool(in_channels=Gs.x.shape[1], hidden_channels=512, out_channels=256).to(device)
    elif backbone == 'sage_maxpool':
        encoder = GraphSAGEEncoderMaxPool(in_channels=Gs.x.shape[1], hidden_channels=512, out_channels=256).to(device)
    else:
        raise ValueError("未知的backbone类型")
    mlp = MLP(in_features=512, hidden_features=256).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(mlp.parameters()), lr=0.0005)
    loss_fn = nn.BCELoss()
    train_data = torch.tensor(training_data_new, dtype=torch.long).to(device)
    batch_size = 128
    dataset = TensorDataset(train_data[0], train_data[1], train_data[2])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_epochs = 500
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0)
    for epoch in epoch_pbar:
        encoder.train()
        mlp.train()
        total_loss = 0
        batch_pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}', position=1, leave=False)
        for idx_Gs, idx_Gt, labels in batch_pbar:
            optimizer.zero_grad()
            idx_Gs = idx_Gs.to(device)
            idx_Gt = idx_Gt.to(device)
            labels = labels.to(device)
            emb_Gs = encoder(Gs_new.x.to(device), Gs_new.edge_index.to(device))
            emb_Gt = encoder(Gt_new.x.to(device), Gt_new.edge_index.to(device))
            node_emb1 = emb_Gs[idx_Gs]
            node_emb2 = emb_Gt[idx_Gt]
            pred = mlp(node_emb1, node_emb2).squeeze()
            loss = loss_fn(pred, labels.float())
            loss.backward()
            optimizer.step()
            del emb_Gs, emb_Gt
            torch.cuda.empty_cache()
            total_loss += loss.item()
            batch_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        avg_loss = total_loss / len(dataloader)
        epoch_pbar.set_postfix({'Avg Loss': f'{avg_loss:.4f}'})
    test_data = torch.tensor(testing_data, dtype=torch.long).to(device)
    CA_path_Gs = "ca_node_embeddings_Gs.json"
    CA_path_Gt = "ca_node_embeddings_Gt.json"
    acc, auc, pred_ = evaluate_model(encoder, mlp, Gs_new, Gt_new, test_data, batch_size, CA_path_Gs, CA_path_Gt)
    print(f"Clean Accuracy on testing data: {acc:.4f}, AUC-ROC: {auc:.4f}")
    test_pos_pairs = [(testing_data[0, i], testing_data[1, i]) for i in range(len(testing_data[0])) if testing_data[2, i] == 1]
    Gs_test_new, Gt_test_new = Gs_new, Gt_new
    with torch.no_grad():
        for u, v in test_pos_pairs:
            u_feature = Gs_test_new.x[u].unsqueeze(0).to(device)
            v_feature = Gt_test_new.x[v].unsqueeze(0).to(device)
            trigger_features_s = gen_s.generate_trigger_features(u_feature, beta)
            trigger_features_t = gen_t.generate_trigger_features(v_feature, beta)
            Gs_test_new, test_trg_s = inject_trigger(Gs_test_new, u, trigger_features_s)
            Gt_test_new, test_trg_t = inject_trigger(Gt_test_new, v, trigger_features_t)
    attacked_path_Gs = "atk_node_embeddings_Gs.json"
    attacked_path_Gt = "atk_node_embeddings_Gt.json"
    acc_,auc_,pred = evaluate_model(encoder, mlp, Gs_test_new, Gt_test_new,test_data,batch_size,attacked_path_Gs,attacked_path_Gt)
    print(acc_,auc_)
    true_labels = testing_data[2]
    _false_indices = np.logical_and(pred == 0, true_labels == 1)
    asr_false_ = np.sum(_false_indices) / np.sum(true_labels == 1)
    print(f"正样本攻击成功率(ASR): {asr_false_:.4f}")
    folder_name_out = f"{dataset}-result/attack_results/t_{similarity_threshold}_alpha_{alpha}_beta_{beta}_ratio_{ratio}_backbone_{backbone}"
    if not os.path.exists(folder_name_out):
        os.makedirs(folder_name_out)
    with open(os.path.join(folder_name_out, f'attack_result.txt'), 'w') as file:
        file.write(f"Clean Accuracy on testing data: {acc:.4f}, AUC-ROC: {auc:.4f}\n")
        file.write(f"正样本攻击成功率（ASR）: {asr_false_:.4f} \n") 