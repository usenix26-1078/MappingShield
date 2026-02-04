import numpy as np


def generate_labeled_test_pairs(pos_pairs, test_pairs, num_nodes1, num_nodes2):
    # 原始测试集取500个，添加标签1
    test_pairs = test_pairs[:test_size]
    labeled_test_pairs = np.hstack((test_pairs, np.ones((test_pairs.shape[0], 1), dtype=int)))

    # 将所有正样本对和测试集对组合，避免重复
    existing_pairs = set(map(tuple, np.vstack((pos_pairs, test_pairs))))

    # 随机生成负样本对
    negative_pairs = []
    while len(negative_pairs) < test_pairs.shape[0]:
        x = np.random.randint(0, num_nodes1)
        y = np.random.randint(0, num_nodes2)
        if (x, y) not in existing_pairs:
            negative_pairs.append((x, y))
            existing_pairs.add((x, y))  # 确保不重复

    # 将负样本对转为数组，并添加标签0
    negative_pairs = np.array(negative_pairs)
    labeled_negative_pairs = np.hstack((negative_pairs, np.zeros((negative_pairs.shape[0], 1), dtype=int)))

    # 合并正负样本对
    new_test_pairs = np.vstack((labeled_test_pairs, labeled_negative_pairs))
    return new_test_pairs


def generate_labeled_train_pairs(pos_pairs, test_pairs, num_nodes1, num_nodes2):
    # 原正样本对添加标签1
    labeled_positive_pairs = np.hstack((pos_pairs, np.ones((pos_pairs.shape[0], 1), dtype=int)))

    # 将所有正样本对和测试集对组合，避免重复
    existing_pairs = set(map(tuple, np.vstack((pos_pairs, test_pairs))))

    # 随机生成负样本对
    negative_pairs = []
    while len(negative_pairs) < pos_pairs.shape[0]:
        x = np.random.randint(0, num_nodes1)
        y = np.random.randint(0, num_nodes2)
        if (x, y) not in existing_pairs:
            negative_pairs.append((x, y))
            existing_pairs.add((x, y))  # 确保不重复

    # 将负样本对转为数组，并添加标签0
    negative_pairs = np.array(negative_pairs)
    labeled_negative_pairs = np.hstack((negative_pairs, np.zeros((negative_pairs.shape[0], 1), dtype=int)))

    # 合并正负样本对
    new_train_pairs = np.vstack((labeled_positive_pairs, labeled_negative_pairs))
    return new_train_pairs


# 加载 .npz 文件
data = np.load(r"./datasets/ACM-DBLP_0.2_updated.npz")
# print(len(data['x1']), len(data['x2']))
# for key in list(data.keys()):
#     print(key, ':', len(data[key]))

test_size = 500
train_size = data["pos_pairs"].shape[0]
print(train_size)



# 计算图1的节点数
num_nodes1 = data["x1"].shape[0]
# 计算图2的节点数
num_nodes2 = data["x2"].shape[0]



# 示例使用
new_test_pairs = generate_labeled_test_pairs(data['pos_pairs'], data['test_pairs'], num_nodes1, num_nodes2)
# print(new_test_pairs)
# print(data["test_pairs"])
filtered_data = new_test_pairs[:, :2] 

# 训练集负样本不包含现有的test_pairs即可

new_train_pairs = generate_labeled_train_pairs(data['pos_pairs'], filtered_data, num_nodes1, num_nodes2)

# 将新的数据存回字典
data_dict = dict(data)  # 将原来的 npz 数据转换为字典
data_dict["new_test_pairs"] = new_test_pairs
data_dict["new_train_pairs"] = new_train_pairs
# data_dict["test_pair_attack"] = test_pair_attack

# 保存为新的 .npz 文件
new_file_path = r"./datasets/ACM-DBLP_0.2_0224.npz"
np.savez(new_file_path, **data_dict)

# 验证保存是否成功
print(f"新的 .npz 文件已保存到: {new_file_path}")

print(type(data_dict['edge_index1']))