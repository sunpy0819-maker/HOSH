"""
实验6: 时序SIR传播实验
选取Top-K节点作为初始感染源，追踪不同时间步的感染规模
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
# 1. 时序 SIR 传播模型
# ==========================================
def run_sir_temporal(graph, seeds, beta, gamma, max_steps=100):
    """
    运行SIR模拟并记录每个时间步的感染规模
    
    返回:
        list: 每个时间步的感染节点总数（包括已恢复的）
    """
    infected_nodes = set(seeds)
    recovered_nodes = set()
    
    # 验证种子节点
    valid_seeds = [n for n in infected_nodes if graph.has_node(n)]
    infected_nodes = set(valid_seeds)
    
    if not infected_nodes:
        return [0] * max_steps
    
    # 记录每个时间步的累计感染规模
    temporal_infection = []
    
    for step in range(max_steps):
        # 记录当前累计感染人数（感染中 + 已恢复）
        total_infected = len(infected_nodes) + len(recovered_nodes)
        temporal_infection.append(total_infected)
        
        if not infected_nodes:
            # 如果没有活跃感染者，后续时间步保持不变
            for _ in range(step + 1, max_steps):
                temporal_infection.append(total_infected)
            break
        
        new_infected = set()
        new_recovered = set()
        
        # 传播过程
        for u in list(infected_nodes):
            # 感染邻居
            for v in graph.neighbors(u):
                if v not in infected_nodes and v not in recovered_nodes:
                    if random.random() < beta:
                        new_infected.add(v)
            
            # 恢复过程
            if random.random() < gamma:
                new_recovered.add(u)
        
        # 更新状态
        infected_nodes.update(new_infected)
        infected_nodes.difference_update(new_recovered)
        recovered_nodes.update(new_recovered)
    
    return temporal_infection

def exp_temporal_sir(methods, g, top_k=10, network_name=None):
    """时序SIR传播实验"""
    print("  [Exp: Temporal SIR] Running propagation analysis...")
    
    N = g.number_of_nodes()
    degrees = [d for n, d in g.degree()]
    k_mean = np.mean(degrees)
    k2_mean = np.mean([d ** 2 for d in degrees])
    
    # 传播阈值: β_th = <k> / (<k²> - <k>)
    beta_th = k_mean / (k2_mean - k_mean)
    # 实际感染率为 1.5 倍传播阈值
    beta = 1.5 * beta_th
    gamma = 0.5
    
    # 使用固定数量的初始感染节点
    top_k = min(top_k, N)  # 确保不超过网络规模
    
    print(f"    Graph Properties: <k>={k_mean:.2f}, <k^2>={k2_mean:.2f}")
    print(f"    Epidemic Threshold: β_th={beta_th:.4f}")
    print(f"    SIR Parameters: beta={beta:.4f} (1.5×β_th), gamma={gamma:.2f}")
    print(f"    Seed Count: {top_k} nodes ({top_k/N*100:.2f}% of network)")
    
    # 统一使用1000次重复，根据网络规模调整时间步
    repeat_times = 1000
    if N < 500:
        max_steps = 50
    elif N > 3000:
        max_steps = 100
    else:
        max_steps = 50
    
    print(f"    Simulation: {repeat_times} repeats, {max_steps} time steps")
    
    # 尝试加载预计算排名
    precomputed = load_precomputed_rankings(network_name) if network_name else None
    
    results = {}
    
    for method_name in methods:
        print(f"    Evaluating: {method_name}")
        
        # 获取节点排名（优先使用预计算数据）
        if precomputed and method_name in precomputed and precomputed[method_name]:
            print(f"      ✓ Using precomputed rankings")
            scores = precomputed[method_name]
        else:
            print(f"      ⚠ Computing on-the-fly")
            scores = get_node_scores(method_name, g)
        ranked_nodes = sorted(scores, key=scores.get, reverse=True)
        
        # 选取Top-K节点作为种子
        seeds = ranked_nodes[:top_k]
        
        # 多次模拟取平均
        temporal_results = np.zeros(max_steps)
        
        for _ in tqdm(range(repeat_times), desc=f"    {method_name}", leave=False):
            infection_curve = run_sir_temporal(g, seeds, beta, gamma, max_steps)
            temporal_results += np.array(infection_curve)
        
        # 计算平均感染率（百分比）
        avg_infection_rate = (temporal_results / repeat_times) / N * 100
        results[method_name] = avg_infection_rate
    
    return results, max_steps, top_k

# ==========================================
# 2. 绘图函数
# ==========================================
def plot_temporal_sir(net, g, temporal_results, max_steps, methods, colors, markers, output_dir, top_k):
    """绘制时序SIR传播曲线（带局部放大图）"""
    
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    time_steps = np.arange(max_steps)
    
    for m in methods:
        if m not in temporal_results:
            continue
        
        # HOSH 使用更粗的线和更大的标记突出显示 - 与 SIR 图一致
        lw = 1.6 if m == 'HOSH' else 1.2
        z_order = 10 if m == 'HOSH' else 5
        ms = 4.5 if m == 'HOSH' else 3.8
        mew = 1.2 if m == 'HOSH' else 0.9
        alpha_val = 0.95 if m == 'HOSH' else 0.88
        
        # 每隔几个步数显示标记点
        markevery = max(3, max_steps // 10)
        
        ax.plot(time_steps, temporal_results[m],
                label=m,
                color=colors.get(m, '#000000'),
                linewidth=lw,
                linestyle='--',
                marker=markers.get(m, 'o'),
                markersize=ms,
                markerfacecolor=colors.get(m, '#000000'),
                markeredgewidth=0.5,
                markeredgecolor='black',
                markevery=markevery,
                zorder=z_order,
                alpha=alpha_val)
    
    ax.set_xlabel("$t$", fontsize=11)
    ax.set_ylabel("$F(t)$ (%)", fontsize=11)
    
    # 不显示标题 - 图表说明应在 caption 中
    
    # 设置坐标轴范围
    ax.set_xlim(-1, max_steps + 1)
    
    # Y轴范围自动调整
    all_values = [val for m in methods if m in temporal_results for val in temporal_results[m]]
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        y_bottom = max(0, y_min - y_range * 0.08)
        y_top = min(100, y_max + y_range * 0.08)
        ax.set_ylim(y_bottom, y_top)
    
    # 图例 - 与 SIR 图保持一致的样式
    ax.legend(loc='lower right', 
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
    
    # 添加局部放大图（嵌入式子图）
    # 找到曲线重叠最严重的区域（方差最小的区域）
    # 计算每个时间步所有方法结果的标准差
    std_per_step = []
    for t in range(max_steps):
        values_at_t = [temporal_results[m][t] for m in methods if m in temporal_results]
        std_per_step.append(np.std(values_at_t))
    
    # 找到标准差最小的连续区间（固定窗口大小确保一致性）
    # 使用固定窗口大小：max_steps的25%，但至少10步，最多20步
    window_size = max(10, min(20, int(max_steps * 0.25)))
    min_std_sum = float('inf')
    best_start = 0
    
    # 确保有足够的数据点进行搜索
    if max_steps > window_size:
        for start in range(max_steps - window_size):
            window_std_sum = sum(std_per_step[start:start + window_size])
            if window_std_sum < min_std_sum:
                min_std_sum = window_std_sum
                best_start = start
    
    zoom_start = best_start
    zoom_end = best_start + window_size
    
    # 创建嵌入式子图 - 位置在左下角，尺寸适中
    axins = ax.inset_axes([0.30, 0.25, 0.36, 0.33])  # [x, y, width, height] in axes coordinates
    
    for m in methods:
        if m not in temporal_results:
            continue
        
        lw = 1.3 if m == 'HOSH' else 0.9
        z_order = 10 if m == 'HOSH' else 5
        alpha_val = 0.95 if m == 'HOSH' else 0.88
        
        # 放大图中不显示标记，只显示线条
        axins.plot(time_steps[zoom_start:zoom_end], temporal_results[m][zoom_start:zoom_end],
                   color=colors.get(m, '#000000'),
                   linewidth=lw,
                   linestyle='--',
                   zorder=z_order,
                   alpha=alpha_val)
    
    # 放大图的范围 - 使用对称的边距避免左右不对称
    x_margin = 0.5  # 左右各减少0.5步的边距
    axins.set_xlim(zoom_start - x_margin, zoom_end + x_margin)
    
    # 放大图Y轴范围 - 使用更激进的放大策略，统一边距比例
    zoom_values = [val for m in methods if m in temporal_results 
                   for val in temporal_results[m][zoom_start:zoom_end]]
    if zoom_values:
        zoom_y_min = min(zoom_values)
        zoom_y_max = max(zoom_values)
        zoom_y_range = zoom_y_max - zoom_y_min
        
        # 更激进的放大策略：
        # 1. 使用更小的相对边距（2%而非5%）
        # 2. 设置最小绝对边距为数据范围的10%（而非固定0.1）
        # 3. 如果数据范围太小（<0.5），使用固定边距0.05
        if zoom_y_range < 0.5:
            # 数据范围极小，使用固定的小边距
            margin = 0.05
        else:
            # 数据范围正常，使用相对边距，确保至少是范围的10%
            relative_margin = zoom_y_range * 0.02
            min_margin = zoom_y_range * 0.10
            margin = max(relative_margin, min_margin)
        
        axins.set_ylim(max(0, zoom_y_min - margin), 
                       min(100, zoom_y_max + margin))
    
    # 放大图刻度设置 - 更小的字体
    axins.tick_params(labelsize=7, direction='out', length=2.0, width=0.6)
    
    # 放大图边框
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color('#000000')
    
    # 手动绘制连接框，避免indicate_inset_zoom的连接线问题
    from matplotlib.patches import Rectangle, ConnectionPatch
    
    # 在主图上绘制放大区域的矩形框
    rect = Rectangle((zoom_start, axins.get_ylim()[0]), 
                     zoom_end - zoom_start, 
                     axins.get_ylim()[1] - axins.get_ylim()[0],
                     fill=False, edgecolor='gray', linewidth=0.8, 
                     linestyle='--', alpha=0.6, transform=ax.transData,
                     zorder=1)
    ax.add_patch(rect)
    
    # 手动绘制连接线（从主图矩形框到子图）
    # 左下角连接
    con1 = ConnectionPatch(
        xyA=(zoom_start, axins.get_ylim()[0]), coordsA=ax.transData,
        xyB=(0, 0), coordsB=axins.transAxes,
        linestyle='--', linewidth=0.7, color='gray', alpha=0.5, zorder=1
    )
    # 右上角连接
    con2 = ConnectionPatch(
        xyA=(zoom_end, axins.get_ylim()[1]), coordsA=ax.transData,
        xyB=(1, 1), coordsB=axins.transAxes,
        linestyle='--', linewidth=0.7, color='gray', alpha=0.5, zorder=1
    )
    ax.add_artist(con1)
    ax.add_artist(con2)
    
    # 边框设置 - 保留所有四条边，与 SIR 图一致
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('#000000')
    
    # 刻度朝外 - 与 SIR 图一致
    ax.tick_params(direction='out', which='major', length=3.0, width=0.7)
    
    plt.tight_layout(pad=0.2)
    
    # 保存图像 - 与 SIR 图相同的参数
    save_path = os.path.join(output_dir, f"Temporal_SIR_{net}.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pil_kwargs={'quality': 95})
    
    print(f"    [Output] High-quality figure saved: {save_path} (600 dpi)")
    plt.close()

# ==========================================
# 3. 数据导出函数
# ==========================================
def save_temporal_results_to_excel(all_results, methods, output_dir):
    """保存时序SIR结果到Excel"""
    print("\n  [Export] Saving Temporal SIR results to Excel...")
    
    for net, data in all_results.items():
        temporal_results = data['temporal_results']
        max_steps = data['max_steps']
        
        # 构建数据框
        df_data = {'Time_Step': np.arange(max_steps)}
        for m in methods:
            if m in temporal_results:
                df_data[f'{m}_Infection_%'] = temporal_results[m]
        
        df = pd.DataFrame(df_data)
        
        # 保存到Excel
        excel_path = os.path.join(output_dir, f"Temporal_Data_{net}.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Temporal_Results', index=False)
            worksheet = writer.sheets['Temporal_Results']
            
            # 调整列宽
            worksheet.column_dimensions['A'].width = 12
            for i, m in enumerate(methods):
                col_letter = chr(ord('B') + i)
                worksheet.column_dimensions[col_letter].width = 15
        
        print(f"    Saved: {excel_path}")

# ==========================================
# 4. 主流程
# ==========================================
def main():
    print("=" * 60)
    print(" Experiment: Temporal SIR Propagation (10 Seeds)")
    print("=" * 60)
    
    set_seed(42)
    
    # 输出目录
    output_dir = "results/exp_temporal_sir"
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
    
    top_k = 10  # 固定使用10个节点作为初始感染源
    
    # 存储所有网络的结果
    all_results = {}
    
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
            
            temporal_results, max_steps, actual_k = exp_temporal_sir(methods, g, top_k, net)
            plot_temporal_sir(net, g, temporal_results, max_steps, methods, colors, markers, output_dir, actual_k)
            
            # 保存结果用于后续导出
            all_results[net] = {
                'temporal_results': temporal_results,
                'max_steps': max_steps
            }
            
        except Exception as e:
            print(f"  [Error] Failed to process {net}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 导出所有结果到Excel
    if all_results:
        save_temporal_results_to_excel(all_results, methods, output_dir)
    
    print("\n" + "=" * 60)
    print(" Experiment: Temporal SIR Completed!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()
