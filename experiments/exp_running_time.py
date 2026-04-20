"""
实验: 运行时间分析
评估不同方法在各个网络上的计算效率
横向柱状图: 纵轴为网络名称，横轴为运行时间(秒)
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

# 导入方法实现
from hosh_methods import get_node_scores
# 导入网络加载模块
from network_loader import download_and_load_graph, get_network_list

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

# ==========================================
# 1. 时间测量函数
# ==========================================
def measure_running_time(method_name, g, repeat_times=20):
    """
    测量方法的平均运行时间
    
    参数:
        method_name: 方法名称
        g: 网络图
        repeat_times: 重复测量次数
    
    返回:
        平均运行时间(秒)
    """
    times = []
    
    for _ in range(repeat_times):
        start_time = time.time()
        try:
            scores = get_node_scores(method_name, g)
            end_time = time.time()
            times.append(end_time - start_time)
        except Exception as e:
            print(f"      [Warning] {method_name} failed: {e}")
            return None
    
    return np.mean(times)

def exp_running_time(methods, networks, colors, output_dir):
    """
    运行时间实验
    
    参数:
        methods: 方法列表
        networks: 网络列表
        colors: 颜色映射
        output_dir: 输出目录
    
    返回:
        results: {network_name: {method_name: time}}
    """
    print("  [Exp: Running Time] Measuring computation efficiency...")
    
    results = {}
    
    for net_idx, net in enumerate(networks, 1):
        print(f"\n  [{net_idx}/{len(networks)}] Network: {net}")
        
        try:
            g = download_and_load_graph(net)
            if g is None or g.number_of_nodes() == 0:
                print(f"    [Skip] Network {net} is empty or failed to load")
                continue
            
            print(f"    Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
            
            results[net] = {}
            
            for method_name in tqdm(methods, desc=f"    Methods for {net}"):
                avg_time = measure_running_time(method_name, g, repeat_times=20)
                
                if avg_time is not None:
                    results[net][method_name] = avg_time
                    print(f"      {method_name}: {avg_time:.4f}s")
                else:
                    results[net][method_name] = None
            
            # 计算完当前网络后立即绘制结果
            plot_single_network_result(net, results[net], methods, colors, output_dir)
                    
        except Exception as e:
            print(f"    [Error] Failed to process {net}: {e}")
            continue
    
    return results

# ==========================================
# 2. 绘图函数
# ==========================================
def plot_single_network_result(net, net_results, methods, colors, output_dir):
    """
    绘制单个网络的运行时间结果
    
    参数:
        net: 网络名称
        net_results: {method_name: time}
        methods: 方法列表
        colors: 颜色映射
        output_dir: 输出目录
    """
    if not net_results:
        print(f"    [Warning] No valid results to plot for {net}")
        return
    
    print(f"    Generating plot for {net}...")
    
    # 创建单独的图
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    # 准备数据
    y_positions = np.arange(len(methods))
    values = []
    
    for method_name in methods:
        time_val = net_results.get(method_name, None)
        if time_val is None:
            values.append(0)
        else:
            values.append(time_val)
    
    # 绘制横向柱状图
    bars = ax.barh(y_positions, values,
                  height=0.6,
                  color=[colors.get(m, '#000000') for m in methods],
                  edgecolor='black',
                  linewidth=0.6,
                  alpha=0.85)
    
    # 在柱子末端添加数值标签
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val > 0:
            # 格式化时间显示 - 根据实际时间自动选择单位
            if val >= 1:
                label_text = f"{val:.3f}s"
            else:
                label_text = f"{val*1000:.1f}ms"
            
            # 在柱子末端添加标签
            x_offset = max(values) * 0.015
            ax.text(bar.get_width() + x_offset, 
                   bar.get_y() + bar.get_height()/2,
                   label_text,
                   va='center',
                   ha='left',
                   fontsize=8)
    
    # 设置Y轴
    ax.set_yticks(y_positions)
    ax.set_yticklabels(methods, fontsize=9)
    ax.invert_yaxis()  # 让第一个方法在顶部
    
    # 设置X轴
    ax.set_xlabel("Running time (s)", fontsize=11)
    
    # X轴范围 - 右侧留出空间给标签
    if max(values) > 0:
        ax.set_xlim(0, max(values) * 1.22)
    
    # 边框设置 - 保留所有四条边
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('#000000')
    
    # 刻度设置 - 朝外
    ax.tick_params(direction='out', which='major', length=3.0, width=0.7)
    
    plt.tight_layout(pad=0.2)
    
    # 保存 PNG 格式
    save_path = os.path.join(output_dir, f"RunningTime_{net}.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pil_kwargs={'quality': 95})
    
    print(f"      Saved: {save_path}")
    plt.close()

def plot_running_time_results(results, methods, colors, output_dir):
    """
    【保留此函数用于兼容性，但实际已在每个网络计算后实时绘制】
    绘制横向柱状图 - 每个网络独立生成一张图
    
    参数:
        results: {network_name: {method_name: time}}
        methods: 方法列表
        colors: 颜色映射
        output_dir: 输出目录
    """
    print("  [Info] All plots have been generated during computation.")
    pass

# ==========================================
# 3. 主流程
# ==========================================
def main():
    print("=" * 60)
    print(" Experiment: Running Time Analysis")
    print("=" * 60)
    
    # 输出目录
    output_dir = "results/exp_running_time"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用统一的网络列表
    networks = get_network_list()
    
    # 统一的方法列表（与改进率实验一致）
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
    
    print(f"\nNetworks to analyze: {networks}")
    print(f"Methods to compare: {methods}")
    
    # 运行时间实验（已包含实时绘图）
    results = exp_running_time(methods, networks, colors, output_dir)
    
    # 绘制结果（已在实验过程中完成）
    plot_running_time_results(results, methods, colors, output_dir)
    
    # 保存数值结果
    result_file = os.path.join(output_dir, "running_time_results.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("Running Time Analysis Results\n")
        f.write("=" * 60 + "\n\n")
        
        for net in results:
            f.write(f"Network: {net}\n")
            f.write("-" * 40 + "\n")
            for method_name in methods:
                time_val = results[net].get(method_name, None)
                if time_val is not None:
                    f.write(f"  {method_name:12s}: {time_val:8.4f}s\n")
                else:
                    f.write(f"  {method_name:12s}: N/A\n")
            f.write("\n")
    
    print(f"\n  [Output] Results saved: {result_file}")
    
    print("\n" + "=" * 60)
    print(" Experiment: Running Time Completed!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()
