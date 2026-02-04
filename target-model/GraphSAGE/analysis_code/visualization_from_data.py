import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.manifold import TSNE
import matplotlib as mpl

# 设置matplotlib字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def set_plot_style():
    """设置绘图样式"""
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

def plot_embedding_changes():
    """绘制节点嵌入变化柱状图"""
    print("正在绘制节点嵌入变化柱状图...")
    
    # 加载t-SNE相关数据
    try:
        selected_embeddings = np.load('tsne_selected_embeddings.npy')
        selected_labels = json.load(open('tsne_selected_labels.json', 'r'))
        selected_pairs = json.load(open('tsne_selected_pairs.json', 'r'))
        
        # 计算相似度变化
        similarities_before = []
        similarities_after = []
        
        # 从嵌入数据中提取相似度信息
        # 假设数据是按照 [Gs-clean, Gt-clean, Gs-trigger, Gt-trigger] 的顺序存储的
        num_samples = len(selected_pairs) // 4  # 每个样本有4个嵌入
        
        for i in range(num_samples):
            # 获取原始节点对的嵌入
            gs_clean_idx = i * 4
            gt_clean_idx = i * 4 + 1
            gs_trigger_idx = i * 4 + 2
            gt_trigger_idx = i * 4 + 3
            
            # 计算原始相似度
            sim_before = np.dot(selected_embeddings[gs_clean_idx], selected_embeddings[gt_clean_idx]) / \
                        (np.linalg.norm(selected_embeddings[gs_clean_idx]) * np.linalg.norm(selected_embeddings[gt_clean_idx]))
            
            # 计算触发器后相似度
            sim_after = np.dot(selected_embeddings[gs_trigger_idx], selected_embeddings[gt_trigger_idx]) / \
                       (np.linalg.norm(selected_embeddings[gs_trigger_idx]) * np.linalg.norm(selected_embeddings[gt_trigger_idx]))
            
            similarities_before.append(sim_before)
            similarities_after.append(sim_after)
        
        # 绘制柱状图
        plt.figure(figsize=(8, 8))
        x = np.arange(num_samples)
        width = 0.35
        
        bar1 = plt.bar(x - width/2, similarities_before, width, label='Clean', color='#45c3af')
        bar2 = plt.bar(x + width/2, similarities_after, width, label='Triggered', color='#021e30')
        
        mean_before = np.mean(similarities_before)
        mean_after = np.mean(similarities_after)
        
        l1 = plt.axhline(y=mean_before, color='#45c3af', linestyle='--', alpha=0.7, linewidth=2.5, 
                        label=f'Clean Mean: {mean_before:.3f}')
        l2 = plt.axhline(y=mean_after, color='#021e30', linestyle='--', alpha=0.7, linewidth=2.5, 
                        label=f'Triggered Mean: {mean_after:.3f}')
        
        plt.xticks(x, [f'{i+1}' for i in range(num_samples)], fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
        
        main_legend = plt.legend([bar1, bar2], ['Clean', 'Triggered'], 
                                loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=True, fontsize=18)
        plt.gca().add_artist(main_legend)
        plt.legend([l1, l2], [l1.get_label(), l2.get_label()], 
                  loc='upper right', frameon=True, fontsize=18)
        
        plt.tight_layout()
        plt.savefig('embedding_changes_from_data.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("节点嵌入变化柱状图已保存为 embedding_changes_from_data.png")
        
    except FileNotFoundError as e:
        print(f"缺少数据文件: {e}")
    except Exception as e:
        print(f"绘制节点嵌入变化图时出错: {e}")

def plot_tsne_visualization():
    """绘制t-SNE可视化图"""
    print("正在绘制t-SNE可视化图...")
    
    try:
        # 加载t-SNE数据
        embeddings_2d = np.load('tsne_embeddings_2d.npy')
        selected_labels = json.load(open('tsne_selected_labels.json', 'r'))
        
        plt.figure(figsize=(10, 8))
        
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
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1, frameon=True, fontsize=14, borderaxespad=0.2)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
        plt.tight_layout()
        plt.savefig('tsne_visualization_from_data.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("t-SNE可视化图已保存为 tsne_visualization_from_data.png")
        
    except FileNotFoundError as e:
        print(f"缺少数据文件: {e}")
    except Exception as e:
        print(f"绘制t-SNE可视化图时出错: {e}")

def plot_logits_changes():
    """绘制Logits变化柱状图"""
    print("正在绘制Logits变化柱状图...")
    
    try:
        # 加载logits数据
        selected_logits_before = np.load('selected_logits_before.npy')
        selected_logits_after = np.load('selected_logits_after.npy')
        neg_logits = np.load('neg_logits.npy')
        
        num_samples = len(selected_logits_before)
        
        # 计算均值
        neg_mean = np.mean(neg_logits)
        pos_mean_before = np.mean(selected_logits_before)
        pos_mean_after = np.mean(selected_logits_after)
        
        # 绘制柱状图
        plt.figure(figsize=(8, 8))
        x = np.arange(num_samples)
        width = 0.35
        
        bar1 = plt.bar(x - width/2, selected_logits_before, width, label='Clean', color='#45c3af')
        bar2 = plt.bar(x + width/2, selected_logits_after, width, label='Triggered', color='#021e30')
        
        l3 = plt.axhline(y=neg_mean, color='#963f5e', linestyle='--', alpha=0.7, linewidth=2.5, 
                        label=f'Neg Mean: {neg_mean:.3f}')
        l1 = plt.axhline(y=pos_mean_before, color='#45c3af', linestyle='--', alpha=0.7, linewidth=2.5, 
                        label=f'Clean Mean: {pos_mean_before:.3f}')
        l2 = plt.axhline(y=pos_mean_after, color='#021e30', linestyle='--', alpha=0.7, linewidth=2.5, 
                        label=f'Triggered Mean: {pos_mean_after:.3f}')
        
        plt.xticks(x, [f'{i+1}' for i in range(num_samples)], fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
        
        main_legend = plt.legend([bar1, bar2], ['Clean', 'Triggered'], 
                                loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=True, fontsize=18)
        plt.gca().add_artist(main_legend)
        plt.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()], 
                  loc='upper right', frameon=True, fontsize=18)
        
        plt.tight_layout()
        plt.savefig('logits_changes_from_data.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Logits变化柱状图已保存为 logits_changes_from_data.png")
        
    except FileNotFoundError as e:
        print(f"缺少数据文件: {e}")
    except Exception as e:
        print(f"绘制Logits变化图时出错: {e}")

def plot_layer_activations():
    """绘制层激活热图"""
    print("正在绘制层激活热图...")
    
    try:
        # 加载激活数据
        activations_before = np.load('activations_before.npy')
        activations_after = np.load('activations_after.npy')
        activation_diffs = np.load('activation_diffs.npy')
        
        num_layers, num_samples = activations_before.shape
        
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
        plt.savefig('layer_activations_before_from_data.png', dpi=300, bbox_inches='tight')
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
        plt.savefig('layer_activations_after_from_data.png', dpi=300, bbox_inches='tight')
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
        plt.savefig('layer_activations_diff_from_data.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("层激活热图已保存")
        
    except FileNotFoundError as e:
        print(f"缺少数据文件: {e}")
    except Exception as e:
        print(f"绘制层激活热图时出错: {e}")

def plot_residual_activations():
    """绘制残差激活热图"""
    print("正在绘制残差激活热图...")
    
    try:
        # 加载残差数据
        residual_before = np.load('residual_before.npy')
        residual_after = np.load('residual_after.npy')
        residual_diffs = np.load('residual_diffs.npy')
        
        num_layers, num_samples = residual_before.shape
        
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
        plt.savefig('residual_activations_before_from_data.png', dpi=300, bbox_inches='tight')
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
        plt.savefig('residual_activations_after_from_data.png', dpi=300, bbox_inches='tight')
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
        plt.savefig('residual_activations_diff_from_data.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("残差激活热图已保存")
        
    except FileNotFoundError as e:
        print(f"缺少数据文件: {e}")
    except Exception as e:
        print(f"绘制残差激活热图时出错: {e}")

def main():
    """主函数：执行所有可视化"""
    print("开始从数据文件重新生成可视化图表...")
    
    # 设置绘图样式
    set_plot_style()
    
    # 检查数据文件是否存在
    required_files = [
        'tsne_selected_embeddings.npy',
        'tsne_embeddings_2d.npy',
        'tsne_selected_labels.json',
        'tsne_selected_pairs.json',
        'selected_logits_before.npy',
        'selected_logits_after.npy',
        'neg_logits.npy',
        'activations_before.npy',
        'activations_after.npy',
        'activation_diffs.npy',
        'residual_before.npy',
        'residual_after.npy',
        'residual_diffs.npy'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("警告：以下数据文件缺失：")
        for file in missing_files:
            print(f"  - {file}")
        print("某些图表可能无法生成。")
    
    # 执行所有可视化
    plot_embedding_changes()
    plot_tsne_visualization()
    plot_logits_changes()
    plot_layer_activations()
    plot_residual_activations()
    
    print("\n所有可视化图表生成完成！")
    print("生成的文件包括：")
    print("- embedding_changes_from_data.png")
    print("- tsne_visualization_from_data.png")
    print("- logits_changes_from_data.png")
    print("- layer_activations_before_from_data.png")
    print("- layer_activations_after_from_data.png")
    print("- layer_activations_diff_from_data.png")
    print("- residual_activations_before_from_data.png")
    print("- residual_activations_after_from_data.png")
    print("- residual_activations_diff_from_data.png")

if __name__ == "__main__":
    main() 