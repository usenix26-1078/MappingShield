import torch
from torch_geometric.data import Data
import numpy as np

def load_graph_from_npz(file_path):
    """
    从 npz 文件中加载图数据，并构建 PyTorch Geometric 的 Data 对象。

    参数:
        file_path (str): npz 文件路径。

    返回:
        Gs (torch_geometric.data.Data): 第一个图。
        Gt (torch_geometric.data.Data): 第二个图。
    """
    # 加载 npz 文件
    data = np.load(file_path, allow_pickle=True)

    # 提取 edge_index 和节点特征
    edge_index1 = torch.tensor(data['edge_index1'], dtype=torch.long)
    edge_index2 = torch.tensor(data['edge_index2'], dtype=torch.long)
    x1 = torch.tensor(data['x1'], dtype=torch.float)
    x2 = torch.tensor(data['x2'], dtype=torch.float)

    # 构建 PyG 的 Data 对象
    Gs = Data(x=x1, edge_index=edge_index1)
    Gt = Data(x=x2, edge_index=edge_index2)

    return Gs, Gt


def load_data_(file_path):
    """
    加载训练数据和新的测试集
    """
    # 加载 npz 文件
    data = np.load(file_path, allow_pickle=True)
    # 新测试集数据
    new_test_pairs = data['new_test_pairs'].astype(np.int64)  # 假设 new_test_pairs 包括节点对和标签
    # 新测试集数据
    new_train_pairs = data['new_train_pairs'].astype(np.int64)  # 假设 new_test_pairs 包括节点对和标签
 

    # 返回值包括新增的数据，同时保留原始接口
    return (
        new_train_pairs.T,
        new_test_pairs.T,
    )



def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    edge_index1, edge_index2 = data['edge_index1'].astype(np.int64), data['edge_index2'].astype(np.int64)
    anchor_links, test_pairs = data['pos_pairs'].astype(np.int64), data['test_pairs'].astype(np.int64)
    x1, x2 = data['x1'].astype(np.float32), data['x2'].astype(np.float32)

    return edge_index1, edge_index2, x1, x2, anchor_links.T, test_pairs.T

