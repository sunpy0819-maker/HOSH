"""
预计算节点排名脚本
提前计算所有网络的所有方法的节点排名分数并保存到本地
实验脚本可以直接加载预计算结果,避免重复计算
"""
import os
import pickle
import numpy as np
from tqdm import tqdm
from network_loader import download_and_load_graph, get_network_list
from hosh_methods import get_node_scores

# 输出目录
OUTPUT_DIR = "results/node_rankings"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 所有方法列表
METHODS = ['HOSH', 'ISH', 'DC', 'BC', 'CC', 'K-Shell', 'SH', 'CI', 'SNC']


def compute_and_save_rankings(network_name, methods=None):
    """
    计算指定网络的所有方法的节点排名分数并保存
    
    参数:
    network_name: 网络名称
    methods: 要计算的方法列表,默认为所有方法
    
    返回:
    dict: {method: {node_id: score}}
    """
    if methods is None:
        methods = METHODS
    
    print(f"\n{'='*60}")
    print(f"正在处理网络: {network_name}")
    print(f"{'='*60}")
    
    # 加载网络
    try:
        g = download_and_load_graph(network_name)
        print(f"✓ 网络加载成功: {g.number_of_nodes()} 个节点, {g.number_of_edges()} 条边")
    except Exception as e:
        print(f"✗ 网络加载失败: {e}")
        return None
    
    # 计算所有方法的分数
    rankings = {}
    
    for method in tqdm(methods, desc=f"计算 {network_name}", ncols=80):
        try:
            scores = get_node_scores(method, g)
            rankings[method] = scores
            print(f"  ✓ {method:10s}: {len(scores)} 个节点")
        except Exception as e:
            print(f"  ✗ {method:10s}: 计算失败 - {e}")
            rankings[method] = None
    
    # 保存结果
    output_file = os.path.join(OUTPUT_DIR, f"{network_name}_rankings.pkl")
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(rankings, f)
        print(f"✓ 结果已保存至: {output_file}")
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        return None
    
    return rankings


def load_precomputed_rankings(network_name):
    """
    加载预计算的节点排名分数
    
    参数:
    network_name: 网络名称
    
    返回:
    dict: {method: {node_id: score}} 或 None (如果文件不存在)
    """
    file_path = os.path.join(OUTPUT_DIR, f"{network_name}_rankings.pkl")
    
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as f:
            rankings = pickle.load(f)
        return rankings
    except Exception as e:
        print(f"✗ 加载 {network_name} 的预计算结果失败: {e}")
        return None


def precompute_all_networks(networks=None, methods=None, force_recompute=False):
    """
    预计算所有网络的节点排名分数
    
    参数:
    networks: 要计算的网络列表,默认为所有可用网络
    methods: 要计算的方法列表,默认为所有方法
    force_recompute: 是否强制重新计算(即使已有缓存)
    """
    if networks is None:
        networks = get_network_list()
    
    if methods is None:
        methods = METHODS
    
    print(f"\n{'#'*70}")
    print(f"# 预计算节点排名任务")
    print(f"# 网络数量: {len(networks)}")
    print(f"# 方法列表: {', '.join(methods)}")
    print(f"# 强制重算: {'是' if force_recompute else '否'}")
    print(f"{'#'*70}\n")
    
    results_summary = {}
    
    for i, network in enumerate(networks, 1):
        print(f"\n[{i}/{len(networks)}] 处理网络: {network}")
        
        # 检查是否已有缓存
        if not force_recompute:
            cached = load_precomputed_rankings(network)
            if cached is not None:
                print(f"✓ 发现缓存结果,跳过计算")
                results_summary[network] = "缓存"
                continue
        
        # 计算并保存
        rankings = compute_and_save_rankings(network, methods)
        
        if rankings is not None:
            results_summary[network] = "成功"
        else:
            results_summary[network] = "失败"
    
    # 打印汇总
    print(f"\n\n{'='*70}")
    print("预计算任务完成!")
    print(f"{'='*70}")
    print(f"{'网络名称':<20s} {'状态'}")
    print(f"{'-'*70}")
    for network, status in results_summary.items():
        status_symbol = "✓" if status in ["成功", "缓存"] else "✗"
        print(f"{status_symbol} {network:<20s} {status}")
    print(f"{'='*70}\n")


def verify_rankings_file(network_name):
    """
    验证预计算文件的完整性
    
    参数:
    network_name: 网络名称
    
    返回:
    bool: 文件是否完整有效
    """
    rankings = load_precomputed_rankings(network_name)
    
    if rankings is None:
        print(f"✗ {network_name}: 文件不存在")
        return False
    
    print(f"\n检查 {network_name}:")
    all_valid = True
    
    for method in METHODS:
        if method not in rankings:
            print(f"  ✗ {method}: 缺失")
            all_valid = False
        elif rankings[method] is None:
            print(f"  ✗ {method}: 数据为None")
            all_valid = False
        else:
            print(f"  ✓ {method}: {len(rankings[method])} 个节点")
    
    return all_valid


def get_top_k_nodes(rankings, method, k):
    """
    从预计算结果中获取Top-K节点
    
    参数:
    rankings: 预计算的排名字典 {method: {node_id: score}}
    method: 方法名称
    k: 返回前k个节点
    
    返回:
    list: Top-K节点ID列表
    """
    if method not in rankings or rankings[method] is None:
        raise ValueError(f"方法 {method} 的排名数据不存在")
    
    scores = rankings[method]
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [node for node, score in sorted_nodes[:k]]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="预计算节点排名脚本")
    parser.add_argument('--networks', nargs='+', default=None,
                        help='指定要计算的网络列表,默认为所有网络')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='指定要计算的方法列表,默认为所有方法')
    parser.add_argument('--force', action='store_true',
                        help='强制重新计算(即使已有缓存)')
    parser.add_argument('--verify', nargs='+', default=None,
                        help='验证指定网络的预计算文件')
    
    args = parser.parse_args()
    
    # 验证模式
    if args.verify:
        print("\n验证预计算文件:")
        for network in args.verify:
            verify_rankings_file(network)
    else:
        # 预计算模式
        precompute_all_networks(
            networks=args.networks,
            methods=args.methods,
            force_recompute=args.force
        )
