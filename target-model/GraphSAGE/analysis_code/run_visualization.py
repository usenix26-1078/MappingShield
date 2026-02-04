#!/usr/bin/env python3
"""
快速运行攻击分析可视化的脚本
"""

import os
import sys
from visualization_from_data import main

def check_environment():
    """检查运行环境"""
    print("检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 6):
        print("错误：需要Python 3.6或更高版本")
        return False
    
    # 检查必需的库
    required_libs = ['numpy', 'matplotlib', 'seaborn', 'sklearn']
    missing_libs = []
    
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)
    
    if missing_libs:
        print(f"错误：缺少以下Python库：{', '.join(missing_libs)}")
        print("请运行：pip install " + " ".join(missing_libs))
        return False
    
    print("环境检查通过！")
    return True

def check_data_files():
    """检查数据文件是否存在"""
    print("检查数据文件...")
    
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
    
    existing_files = []
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            missing_files.append(file)
    
    print(f"找到 {len(existing_files)} 个数据文件")
    print(f"缺少 {len(missing_files)} 个数据文件")
    
    if missing_files:
        print("缺少的文件：")
        for file in missing_files:
            print(f"  - {file}")
        
        if len(missing_files) == len(required_files):
            print("\n错误：没有找到任何数据文件！")
            print("请确保已经运行了 attack-analysis.py 并生成了数据文件。")
            return False
        else:
            print("\n警告：部分数据文件缺失，某些图表可能无法生成。")
    
    return True

def main_wrapper():
    """主函数包装器"""
    print("=" * 50)
    print("攻击分析可视化工具")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        return
    
    # 检查数据文件
    if not check_data_files():
        return
    
    print("\n开始生成可视化图表...")
    print("-" * 30)
    
    try:
        # 运行可视化
        main()
        print("\n" + "=" * 50)
        print("可视化完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n错误：运行过程中出现异常：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_wrapper() 