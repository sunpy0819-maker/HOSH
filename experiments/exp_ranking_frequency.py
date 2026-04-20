"""
实验11: 排名频率分布图
分析不同方法排名的频率分布
横轴为排名，纵轴为该排名对应的节点数量
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from collections import Counter

# 导入方法实现
from hosh_methods import get_node_scores
# 导入网络加载模块
from network_loader import download_and_load_graph, get_network_list
# 导入预计算排名模块
from precompute_rankings import load_precomputed_rankings

# ==========================================
# 0. 基础配置
# ==========================================

plt.rcParams.update({
    'font.family': 'Times New Roman',  # 改为 Times New Roman
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 4.5,
    'axes.grid': False,
    'axes.spines.top': True,  # 保留所有边框
    'axes.spines.right': True,  # 保留所有边框
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0
})

# ==========================================
# 1. 排名频率分析
# ==========================================
def calculate_ranking_frequency(method_name, g, network_name=None):
    """
    计算某方法下的排名频率分布
    
    返回:
        ranking: 排名位置列表 (1-based)
        frequency: 对应排名的节点数量
    """
    # 尝试加载预计算排名
    precomputed = load_precomputed_rankings(network_name) if network_name else None
    
    # 获取节点分数（优先使用预计算数据）
    if precomputed and method_name in precomputed and precomputed[method_name]:
        scores = precomputed[method_name]
    else:
        scores = get_node_scores(method_name, g)
    
    # 计算排名 (相同分数获得相同排名)
    # 先按分数降序排序
    sorted_scores = sorted(set(scores.values()), reverse=True)
    score_to_rank = {}
    
    current_rank = 1
    for score_value in sorted_scores:
        score_to_rank[score_value] = current_rank
        # 计算有多少个节点有这个分数
        count = sum(1 for s in scores.values() if s == score_value)
        current_rank += count
    
    # 为每个节点分配排名
    node_ranks = {node: score_to_rank[score] for node, score in scores.items()}
    
    # 统计排名频率
    rank_counter = Counter(node_ranks.values())
    
    # 转换为列表形式 (按排名排序)
    rankings = sorted(rank_counter.keys())
    frequencies = [rank_counter[r] for r in rankings]
    
    return rankings, frequencies

def exp_ranking_frequency(methods, g, network_name=None):
    """排名频率分布实验"""
    print("  [Exp: Ranking Frequency] Calculating distribution...")
    
    results = {}
    
    for method_name in methods:
        print(f"    Analyzing: {method_name}")
        rankings, frequencies = calculate_ranking_frequency(method_name, g, network_name)
        results[method_name] = {
            'rankings': rankings,
            'frequencies': frequencies
        }
    
    return results

# ==========================================
# 2. 绘图函数
# ==========================================
def plot_ranking_frequency(net, g, results, methods, colors, markers, output_dir):
    """绘制排名频率分布图"""
    
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    for m in methods:
        if m not in results:
            continue
        
        rankings = results[m]['rankings']
        frequencies = results[m]['frequencies']
        
        # HOSH使用更突出的样式，其他方法使用较小标记
        if m == 'HOSH':
            ms = 3.0      # 减小标记尺寸
            alpha = 0.90
            z_order = 10
        else:
            ms = 2.5      # 其他方法使用更小标记
            alpha = 0.70
            z_order = 5
        
        ax.scatter(rankings, frequencies,
                   label=m,
                   color=colors.get(m, '#000000'),
                   marker=markers.get(m, 'o'),
                   s=ms**2 * 5.5,
                   alpha=alpha,
                   edgecolors='none',
                   linewidths=0,
                   zorder=z_order)
    
    ax.set_xlabel("Ranking", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    
    # 不显示标题 - 图表说明应在 caption 中
    
    # 设置坐标轴范围 - 更精确的计算
    N = g.number_of_nodes()
    all_rankings = [r for m in methods if m in results for r in results[m]['rankings']]
    all_frequencies = [f for m in methods if m in results for f in results[m]['frequencies']]
    
    if all_rankings:
        x_min = min(all_rankings)
        x_max = max(all_rankings)
        x_range = x_max - x_min
        # 添加更小的边距，让数据点更集中
        ax.set_xlim(x_min - x_range*0.03, x_max + x_range*0.03)
    
    if all_frequencies:
        y_min = 0  # Y轴从0开始
        y_max = max(all_frequencies)
        y_range = y_max - y_min
        # 上方留出适当空间给图例
        ax.set_ylim(y_min - y_range*0.02, y_max + y_range*0.15)
    
    # 图例 - 优化布局和标记大小
    legend = ax.legend(loc='upper right', 
              frameon=True, 
              fancybox=False,
              shadow=False, 
              edgecolor='black',
              framealpha=0.90,
              ncol=1, 
              columnspacing=0.5,
              labelspacing=0.25,
              handlelength=1.3,
              handletextpad=0.35,
              borderpad=0.25,
              borderaxespad=0.35,
              fontsize=8)
    
    # 边框设置 - 保留所有四条边
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('#000000')
    
    # 刻度朝外
    ax.tick_params(direction='out', which='major', length=3.0, width=0.7)
    
    plt.tight_layout(pad=0.2)
    
    # 只保存 PNG 格式
    save_path = os.path.join(output_dir, f"RankingFreq_{net}.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pil_kwargs={'quality': 95})
    
    print(f"    [Output] High-quality figure saved: {save_path} (600 dpi)")
    plt.close()

# ==========================================
# 3. 数据导出函数
# ==========================================
def save_ranking_frequency_to_excel(all_results, methods, output_dir):
    """保存排名频率结果到Excel"""
    print("\n  [Export] Saving Ranking Frequency results to Excel...")
    
    for net, results in all_results.items():
        # 为每个方法创建一个sheet
        excel_path = os.path.join(output_dir, f"RankingFreq_Data_{net}.xlsx")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for m in methods:
                if m in results:
                    df_data = {
                        'Ranking': results[m]['rankings'],
                        'Frequency': results[m]['frequencies']
                    }
                    df = pd.DataFrame(df_data)
                    
                    sheet_name = m[:31]  # Excel sheet name限制31字符
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    worksheet = writer.sheets[sheet_name]
                    
                    # 调整列宽
                    worksheet.column_dimensions['A'].width = 12
                    worksheet.column_dimensions['B'].width = 12
        
        print(f"    Saved: {excel_path}")

# ==========================================
# 4. 主流程
# ==========================================
def main():
    print("=" * 60)
    print(" Experiment: Ranking Frequency Distribution")
    print("=" * 60)
    
    # 输出目录
    output_dir = "results/exp_ranking_frequency"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用统一的网络列表
    networks = get_network_list()

    # 统一的方法列表
    methods = ['HOSH', 'ISH', 'DC', 'BC', 'CC', 'K-Shell', 'SH', 'CI', 'SNC']

    colors = {
        'HOSH': '#D63230',      # 红色 - 主要方法突出
        'ISH': '#F08C3D',       # 橙色
        'DC': '#E5B25D',        # 金黄色
        'BC': '#4FA3D1',        # 蓝色
        'CC': '#4364B8',        # 深蓝色
        'K-Shell': '#A855A8',   # 紫色
        'SH': '#E2739F',        # 粉色
        'CI': '#8D6E63',        # 棕色
        'SNC': '#4DB6AC'        # 蓝绿色
    }
    
    markers = {
        'HOSH': 'o',      # 圆形 (最经典)
        'ISH': 's',       # 正方形
        'DC': '^',        # 上三角
        'BC': 'D',        # 菱形
        'CC': 'X',        # 粗叉号
        'K-Shell': 'P',   # 粗加号
        'SH': 'v',        # 下三角
        'CI': 'h',        # 平顶六边形
        'SNC': 'H'        # 尖顶六边形
    }
    
    # 存储所有网络的结果
    all_results = {}

    # 对每个网络进行实验
    for i, net in enumerate(networks, 1):
        print(f"\n[{i}/{len(networks)}] Processing network: {net}")
        print("-" * 60)
        
        try:
            g = download_and_load_graph(net, verbose=False)
            
            if g is None or g.number_of_nodes() == 0:
                print(f"  [Skip] Network {net} is empty or failed to load")
                continue
            
            print(f"  Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
            
            # 计算排名频率分布
            results = exp_ranking_frequency(methods, g, net)
            
            # 绘制单个网络的图
            plot_ranking_frequency(net, g, results, methods, colors, markers, output_dir)
            
            # 保存结果用于后续导出
            all_results[net] = results
            
        except Exception as e:
            print(f"  [Error] Failed to process {net}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 导出所有结果到Excel
    if all_results:
        save_ranking_frequency_to_excel(all_results, methods, output_dir)
    
    print("\n" + "=" * 60)
    print(" Experiment: Ranking Frequency Completed!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()
