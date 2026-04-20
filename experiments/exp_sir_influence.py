"""
实验1: SIR影响力最大化实验
评估不同关键节点识别方法在传播动力学中的表现
"""
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

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
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5
})

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    print(f"[Init] Global seed set to: {seed}")

# ==========================================
# 1. SIR 传播模型
# ==========================================
def run_sir_simulation(graph, seeds, beta, gamma, max_steps=1000):
    """单次 SIR 模拟"""
    infected_nodes = set(seeds)
    recovered_nodes = set()
    valid_seeds = [n for n in infected_nodes if graph.has_node(n)]
    infected_nodes = set(valid_seeds)

    if not infected_nodes:
        return 0

    for _ in range(max_steps):
        if not infected_nodes:
            break

        new_infected = set()
        new_recovered = set()

        for u in list(infected_nodes):
            for v in graph.neighbors(u):
                if v not in infected_nodes and v not in recovered_nodes:
                    if random.random() < beta:
                        new_infected.add(v)

            if random.random() < gamma:
                new_recovered.add(u)

        infected_nodes.update(new_infected)
        infected_nodes.difference_update(new_recovered)
        recovered_nodes.update(new_recovered)

    total_impact = len(recovered_nodes) + len(infected_nodes)
    return total_impact

def exp_influence_maximization(methods, g, network_name=None):
    """SIR影响力最大化实验"""
    print("  [Exp: SIR Influence] Running analysis...")

    N = g.number_of_nodes()
    degrees = [d for n, d in g.degree()]
    k_mean = np.mean(degrees)
    k2_mean = np.mean([d ** 2 for d in degrees])

    # 传播阈值: β_th = <k> / (<k²> - <k>)
    beta_th = k_mean / (k2_mean - k_mean)
    # 实际感染率为 1.5 倍传播阈值
    beta = 1.5 * beta_th
    gamma = 0.5

    print(f"    Graph Properties: <k>={k_mean:.2f}, <k^2>={k2_mean:.2f}")
    print(f"    Epidemic Threshold: β_th={beta_th:.4f}")
    print(f"    SIR Parameters: beta={beta:.4f} (1.5×β_th), gamma={gamma:.2f}")

    # 统一使用1000次重复实验
    repeat_times = 1000
    print(f"    Simulation Repeats: {repeat_times}")

    seed_ratios = np.arange(0.025, 0.275, 0.025)
    results = {}

    # 尝试加载预计算排名
    precomputed = load_precomputed_rankings(network_name) if network_name else None

    for method_name in methods:
        print(f"    Evaluating: {method_name}")
        # 优先使用预计算数据
        if precomputed and method_name in precomputed and precomputed[method_name]:
            print(f"      ✓ Using precomputed rankings")
            scores = precomputed[method_name]
        else:
            print(f"      ⚠ Computing rankings on-the-fly")
            scores = get_node_scores(method_name, g)
        ranked_nodes = sorted(scores, key=scores.get, reverse=True)
        y_values = []

        for ratio in tqdm(seed_ratios, desc=f"    {method_name}", leave=False):
            k = int(N * ratio)
            if k == 0: k = 1
            seeds = ranked_nodes[:k]

            total_infected = 0
            for _ in range(repeat_times):
                total_infected += run_sir_simulation(g, seeds, beta, gamma)

            avg_infected_rate = (total_infected / repeat_times) / N
            y_values.append(avg_infected_rate * 100)

        results[method_name] = y_values

    return seed_ratios, results

# ==========================================
# 2. 绘图函数
# ==========================================
def plot_sir_results(net, g, x_ratios, sir_results, methods, colors, markers, output_dir):
    """绘制 SIR 传播实验结果图"""
    
    fig, ax = plt.subplots(figsize=(3.5, 2.8))  # 单栏图尺寸，高度稍小
    
    for m in methods:
        if m not in sir_results:
            continue
        
        # HOSH 使用更粗的线和更大的标记突出显示
        lw = 1.6 if m == 'HOSH' else 1.2
        z_order = 10 if m == 'HOSH' else 5
        ms = 4.5 if m == 'HOSH' else 3.8
        mew = 1.2 if m == 'HOSH' else 0.9
        alpha_val = 0.95 if m == 'HOSH' else 0.88
        
        ax.plot(x_ratios * 100, sir_results[m],
                label=m,
                color=colors.get(m, '#000000'),
                linewidth=lw,
                linestyle='--',
                marker=markers.get(m, 'o'),
                markersize=ms,
                markerfacecolor=colors.get(m, '#000000'),
                markeredgewidth=0.5,
                markeredgecolor='black',
                markevery=1,
                zorder=z_order,
                alpha=alpha_val)
    
    ax.set_xlabel("$p$ (%)", fontsize=11)
    ax.set_ylabel("$F(t_c)$ (%)", fontsize=11)
    
    # 不显示标题 - 图表说明应在 caption 中
    
    # 设置坐标轴范围 - 左右留出适当空间
    ax.set_xlim(1.7, 25.8)
    
    all_values = [val for m in methods if m in sir_results for val in sir_results[m]]
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        y_bottom = max(0, y_min - y_range * 0.08)
        y_top = min(100, y_max + y_range * 0.08)
        ax.set_ylim(y_bottom, y_top)
    
    # 刻度设置
    ax.set_xticks(np.arange(2.5, 26, 2.5))
    
    # 图例
    ax.legend(loc='upper left', 
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
    save_path = os.path.join(output_dir, f"SIR_{net}.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pil_kwargs={'quality': 95})
    
    print(f"    [Output] High-quality figure saved: {save_path} (600 dpi)")
    plt.close()

# ==========================================
# 3. 数据导出函数
# ==========================================
def save_sir_results_to_excel(all_results, methods, output_dir):
    """保存SIR实验结果到Excel"""
    print("\n  [Export] Saving SIR results to Excel...")
    
    for net, data in all_results.items():
        x_ratios = data['x_ratios']
        sir_results = data['sir_results']
        
        # 构建数据框
        df_data = {'Seed_Ratio_%': x_ratios * 100}
        for m in methods:
            if m in sir_results:
                df_data[m] = sir_results[m]
        
        df = pd.DataFrame(df_data)
        
        # 保存到Excel
        excel_path = os.path.join(output_dir, f"SIR_Data_{net}.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='SIR_Results', index=False)
            worksheet = writer.sheets['SIR_Results']
            
            # 调整列宽
            worksheet.column_dimensions['A'].width = 15
            for i, m in enumerate(methods):
                col_letter = chr(ord('B') + i)
                worksheet.column_dimensions[col_letter].width = 12
        
        print(f"    Saved: {excel_path}")

# ==========================================
# 4. 主流程
# ==========================================
def main():
    print("=" * 60)
    print(" Experiment: SIR Influence Maximization")
    print("=" * 60)
    
    set_seed(42)
    
    # 输出目录
    output_dir = "results/exp_sir_influence"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用统一的网络列表
    networks = get_network_list()

    # 统一的方法列表
    methods = ['HOSH', 'ISH', 'DC', 'BC', 'CC', 'K-Shell', 'SH', 'CI', 'SNC']
    
    # 存储所有网络的结果
    all_results = {}

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
    
    # 标记符号 - 易于区分
    markers = {
        'HOSH': 'o',      # 圆形 (最经典)
        'ISH': 's',       # 正方形
        'DC': '^',        # 上三角
        'BC': 'D',        # 菱形
        'CC': 'X',        # 粗叉号
        'K-Shell': 'P',   # 粗加号
        'SH': 'v',        # 下三角
        'CI': 'h',        # 平顶六边形 (hexagon1)
        'SNC': 'H'        # 尖顶六边形 (hexagon2)
    }

    # 对每个网络进行实验
    for i, net in enumerate(networks, 1):
        print(f"\n[{i}/{len(networks)}] Processing network: {net}")
        print("-" * 60)
        
        try:
            g = download_and_load_graph(net)
            if g is None or g.number_of_nodes() == 0:
                print(f"  [Skip] Network {net} is empty or failed to load")
                continue
            
            print(f"  Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
            
            x_ratios, sir_results = exp_influence_maximization(methods, g, net)
            plot_sir_results(net, g, x_ratios, sir_results, methods, colors, markers, output_dir)
            
            # 保存结果用于后续导出
            all_results[net] = {
                'x_ratios': x_ratios,
                'sir_results': sir_results
            }
            
        except Exception as e:
            print(f"  [Error] Failed to process {net}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 导出所有结果到Excel
    if all_results:
        save_sir_results_to_excel(all_results, methods, output_dir)
    
    print("\n" + "=" * 60)
    print(" Experiment: SIR Influence Completed!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()
