"""
实验: 生成网络分析
在不同规模的合成网络上评估算法的可扩展性和鲁棒性
包括: BA无标度网络、WS小世界网络、ER随机网络
"""
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

# 导入方法实现
from hosh_methods import get_node_scores

# ==========================================
# 0. 基础配置
# ==========================================

plt.rcParams.update({
    'font.family': 'Times New Roman',
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
    'axes.spines.top': True,
    'axes.spines.right': True,
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
# 1. 生成网络函数
# ==========================================
def generate_ba_network(n, m=3):
    """生成BA无标度网络"""
    return nx.barabasi_albert_graph(n, m, seed=42)

def generate_ws_network(n, k=6, p=0.3):
    """生成WS小世界网络"""
    return nx.watts_strogatz_graph(n, k, p, seed=42)

def generate_er_network(n, p=None):
    """生成ER随机网络，保持平均度约为6"""
    if p is None:
        p = 6.0 / n
    return nx.erdos_renyi_graph(n, p, seed=42)

# ==========================================
# 2. 时间测量函数
# ==========================================
def measure_running_time(method_name, g, repeat_times=1):
    """
    测量方法的运行时间
    
    参数:
        method_name: 方法名称
        g: 网络图
        repeat_times: 重复测量次数（默认1次）
    
    返回:
        运行时间(秒)，失败返回None
    """
    times = []
    
    for _ in range(repeat_times):
        start_time = time.time()
        try:
            scores = get_node_scores(method_name, g)
            end_time = time.time()
            elapsed = end_time - start_time
            
            # 只记录有效的正时间值
            if elapsed > 0:
                times.append(elapsed)
        except Exception as e:
            print(f"      [Warning] {method_name} failed: {e}")
            return None
    
    # 如果所有测量都失败或为0，返回None
    if not times:
        return None
    
    return np.mean(times)

def exp_synthetic_networks(methods, network_type, network_sizes):
    """
    生成网络实验
    
    参数:
        methods: 方法列表
        network_type: 网络类型 ('BA', 'WS', 'ER')
        network_sizes: 网络规模列表
    
    返回:
        results: {method_name: [times]}
    """
    print(f"  [Exp: Synthetic Networks - {network_type}] Measuring scalability...")
    
    results = {m: [] for m in methods}
    
    for size in tqdm(network_sizes, desc=f"    Network sizes for {network_type}"):
        print(f"\n    Network size: {size} nodes")
        
        try:
            # 生成网络
            if network_type == 'BA':
                g = generate_ba_network(size, m=3)
            elif network_type == 'WS':
                g = generate_ws_network(size, k=6, p=0.3)
            elif network_type == 'ER':
                g = generate_er_network(size)
            else:
                raise ValueError(f"Unknown network type: {network_type}")
            
            # 确保生成的是连通图，如果不是则取最大连通分量
            if not nx.is_connected(g):
                g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
                print(f"      Taking largest connected component: {g.number_of_nodes()} nodes")
            
            print(f"      Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
            
            # 测量每个方法的运行时间
            for method_name in methods:
                # 对快速方法使用更多重复次数以提高测量精度
                repeat_times = 3
                avg_time = measure_running_time(method_name, g, repeat_times=repeat_times)
                
                if avg_time is not None and avg_time > 0:
                    results[method_name].append(avg_time)
                    print(f"        {method_name}: {avg_time:.4f}s")
                else:
                    # 如果测量失败或时间为0，记录为None
                    results[method_name].append(None)
                    if avg_time == 0:
                        print(f"        {method_name}: Too fast to measure accurately")
                    
        except Exception as e:
            print(f"      [Error] Failed to process size {size}: {e}")
            for method_name in methods:
                results[method_name].append(None)
            continue
    
    return results

# ==========================================
# 3. 绘图函数
# ==========================================
def plot_synthetic_results(network_type, network_sizes, results, methods, colors, markers, output_dir):
    """绘制生成网络实验结果图"""
    
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    for m in methods:
        if m not in results:
            continue
        
        # 过滤掉None值
        valid_data = [(size, time) for size, time in zip(network_sizes, results[m]) if time is not None]
        if not valid_data:
            continue
        
        sizes, times = zip(*valid_data)
        
        # HOSH 使用更粗的线和更大的标记突出显示
        lw = 1.6 if m == 'HOSH' else 1.2
        z_order = 10 if m == 'HOSH' else 5
        ms = 4.5 if m == 'HOSH' else 3.8
        mew = 1.2 if m == 'HOSH' else 0.9
        alpha_val = 0.95 if m == 'HOSH' else 0.88
        
        ax.plot(sizes, times,
                label=m,
                color=colors.get(m, '#000000'),
                linewidth=lw,
                linestyle='--',
                marker=markers.get(m, 'o'),
                markersize=ms,
                markerfacecolor=colors.get(m, '#000000'),
                markeredgewidth=0.5,
                markeredgecolor='black',
                zorder=z_order,
                alpha=alpha_val)
    
    ax.set_xlabel("Network size", fontsize=11)
    ax.set_ylabel("Running time", fontsize=11)
    
    # 不显示标题 - 图表说明应在 caption 中
    
    # 使用双对数坐标
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 设置坐标轴范围
    if network_sizes:
        size_min = min(network_sizes)
        size_max = max(network_sizes)
        ax.set_xlim(size_min * 0.9, size_max * 1.1)
    
    all_values = [t for m in methods if m in results for t in results[m] if t is not None]
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min * 0.5, y_max * 2)
    
    # 格式化y轴刻度标签，使用科学计数法并确保对齐
    from matplotlib.ticker import LogFormatterSciNotation
    formatter = LogFormatterSciNotation(labelOnlyBase=False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(formatter)
    
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
    
    # 保存 PNG 格式
    save_path = os.path.join(output_dir, f"Synthetic_{network_type}.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pil_kwargs={'quality': 95})
    
    print(f"    [Output] High-quality figure saved: {save_path} (600 dpi)")
    plt.close()

# ==========================================
# 4. 主流程
# ==========================================
def main():
    print("=" * 60)
    print(" Experiment: Synthetic Networks Analysis")
    print("=" * 60)
    
    set_seed(42)
    
    # 输出目录
    output_dir = "results/exp_synthetic_networks"
    os.makedirs(output_dir, exist_ok=True)
    
    # 统一的方法列表
    methods = ['HOSH','K-Shell', 'ISH', 'CC','DC' ,'SH','BC', 'CI', 'SNC']
    
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
        'CI': 'h',        # 平顶六边形
        'SNC': 'H'        # 尖顶六边形
    }
    
    # 网络规模设置：采用前密后疏的采样策略
    network_sizes = [1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000]
    
    # 网络类型
    network_types = ['BA', 'WS', 'ER']
    
    print(f"\nNetwork sizes: {network_sizes[0]} to {network_sizes[-1]} (step: 1000)")
    print(f"Network types: {network_types}")
    print(f"Methods to compare: {methods}")
    
    # 对每种网络类型进行实验
    for idx, net_type in enumerate(network_types, 1):
        print(f"\n[{idx}/{len(network_types)}] Processing network type: {net_type}")
        print("-" * 60)
        
        try:
            # 运行实验
            results = exp_synthetic_networks(methods, net_type, network_sizes)
            
            # 立即生成结果图
            plot_synthetic_results(net_type, network_sizes, results, methods, colors, markers, output_dir)
            
            # 保存数值结果
            result_file = os.path.join(output_dir, f"synthetic_{net_type}_results.txt")
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(f"Synthetic Network Analysis Results - {net_type}\n")
                f.write("=" * 60 + "\n\n")
                
                for size in network_sizes:
                    f.write(f"Network size: {size} nodes\n")
                    f.write("-" * 40 + "\n")
                    idx = network_sizes.index(size)
                    for method_name in methods:
                        time_val = results[method_name][idx] if idx < len(results[method_name]) else None
                        if time_val is not None:
                            f.write(f"  {method_name:12s}: {time_val:8.4f}s\n")
                        else:
                            f.write(f"  {method_name:12s}: N/A\n")
                    f.write("\n")
            
            print(f"    [Output] Results saved: {result_file}")
            
        except Exception as e:
            print(f"  [Error] Failed to process {net_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print(" Experiment: Synthetic Networks Completed!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()
