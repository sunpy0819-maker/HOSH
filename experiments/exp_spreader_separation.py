"""
实验14: 传播者分离度分析 (Average Shortest Path Length)
评估不同方法识别的关键节点在网络中的分散程度
"""
import networkx as nx
import numpy as np
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

# ==========================================
# 1. 平均最短路径长度计算
# ==========================================
def calculate_spreader_separation(g, spreaders):
    """
    计算传播者集合的平均最短路径长度
    
    公式: L_s = 1/(|S|(|S|-1)) * Σ_{i,j∈S,i≠j} d_{ij}
    
    参数:
    g: networkx.Graph - 网络图
    spreaders: list - 选定的传播者节点列表
    
    返回:
    float: 平均最短路径长度
    """
    if len(spreaders) <= 1:
        return 0.0
    
    # 确保所有节点都在图中
    valid_spreaders = [s for s in spreaders if g.has_node(s)]
    
    if len(valid_spreaders) <= 1:
        return 0.0
    
    # 获取网络的直径（用于处理不连通的节点对）
    # 对于不连通的节点对，距离设为 δ + 1
    try:
        # 尝试计算直径（仅对连通图有效）
        if nx.is_connected(g):
            diameter = nx.diameter(g)
        else:
            # 对于非连通图，使用最大连通分量的直径
            largest_cc = max(nx.connected_components(g), key=len)
            subgraph = g.subgraph(largest_cc)
            diameter = nx.diameter(subgraph)
    except:
        # 如果出错，使用网络规模的上界估计
        diameter = g.number_of_nodes()
    
    disconnected_distance = diameter + 1
    
    # 计算所有传播者对之间的最短路径长度
    total_distance = 0.0
    pair_count = 0
    
    for i, node_i in enumerate(valid_spreaders):
        for node_j in valid_spreaders[i+1:]:
            try:
                # 尝试计算最短路径长度
                distance = nx.shortest_path_length(g, node_i, node_j)
            except nx.NetworkXNoPath:
                # 如果不连通，使用 δ + 1
                distance = disconnected_distance
            
            total_distance += distance
            pair_count += 1
    
    if pair_count == 0:
        return 0.0
    
    # 计算平均值
    avg_distance = total_distance / pair_count
    
    return avg_distance

def exp_spreader_separation(methods, g, network_name=None):
    """
    传播者分离度实验
    
    评估不同方法选择的top-k节点在网络中的分散程度
    路径长度越长，说明节点越分散，影响范围可能越广
    """
    print("  [Exp: Spreader Separation] Running analysis...")
    
    N = g.number_of_nodes()
    
    # 动态调整采样点
    if N < 500:
        seed_ratios = np.arange(0.0, 0.21, 0.02)
    else:
        seed_ratios = np.arange(0.01, 0.21, 0.02)
    
    # 尝试加载预计算排名
    precomputed = load_precomputed_rankings(network_name) if network_name else None
    
    results = {}
    
    for method_name in methods:
        print(f"    Evaluating: {method_name}")
        # 优先使用预计算数据
        if precomputed and method_name in precomputed and precomputed[method_name]:
            print(f"      ✓ Using precomputed rankings")
            scores = precomputed[method_name]
        else:
            print(f"      ⚠ Computing on-the-fly")
            scores = get_node_scores(method_name, g)
        ranked_nodes = sorted(scores, key=scores.get, reverse=True)
        
        separation_values = []
        
        for ratio in tqdm(seed_ratios, desc=f"    {method_name}", leave=False):
            k = max(2, int(N * ratio))  # 至少需要2个节点才能计算距离
            spreaders = ranked_nodes[:k]
            
            # 计算平均最短路径长度
            avg_distance = calculate_spreader_separation(g, spreaders)
            separation_values.append(avg_distance)
        
        results[method_name] = separation_values
    
    return seed_ratios, results

# ==========================================
# ==========================================
def plot_separation_results(net, g, x_ratios, separation_results, methods, colors, markers, output_dir, fixed_scale=True):
    """
    绘制演化雷达图 (Evolutionary Radar Chart)
    
    设计原则：
    1. 单栏宽度89mm (3.5英寸)
    2. 清晰的视觉层次：HOSH方法突出显示
    3. 极简设计：去除冗余元素，保持科学性
    4. 高对比度：确保黑白打印清晰可读
    5. 专业标注：精确的数值和清晰的图例
    
    雷达轴设计：
    - 5个轴：代表Top-5%, 10%, 15%, 20%, 25%的节点
    - 中心点：L_s = 0（完全重叠）
    - 向外延伸：分离度增大（节点更分散）
    
    参数:
    fixed_scale: bool - 是否使用等比例归一化刻度（推荐True，便于跨网络比较）
    """
    
    # 选择5个关键节点比例作为雷达轴
    target_ratios = [0.05, 0.10, 0.15, 0.20, 0.25]
    radar_labels = ['5%', '10%', '15%', '20%', '25%']
    
    # 找到最接近目标比例的数据点索引
    radar_indices = []
    for target in target_ratios:
        idx = np.argmin(np.abs(x_ratios - target))
        radar_indices.append(idx)
    
    # 提取雷达图数据
    radar_data = {}
    max_value = 0
    min_value = float('inf')
    for m in methods:
        if m in separation_results:
            radar_data[m] = [separation_results[m][i] for i in radar_indices]
            max_value = max(max_value, max(radar_data[m]))
            min_value = min(min_value, min(radar_data[m]))
    
    if not radar_data:
        return
    
    # 记录原始数值范围（用于标注）
    original_min = min_value
    original_max = max_value
    
    # 记录原始数值范围（用于标注）
    original_min = min_value
    original_max = max_value
    
    # 等比例归一化到 0-1 区间（保留相对大小关系）
    if fixed_scale:
        # 归一化：将 [min_value, max_value] 映射到 [0, 1]
        value_range = max_value - min_value
        if value_range > 0:
            for m in radar_data:
                radar_data[m] = [(v - min_value) / value_range for v in radar_data[m]]
        y_max = 1.0
        print(f"      [Scale] {net}: normalized to [0, 1], original range [{original_min:.2f}, {original_max:.2f}]")
    else:
        # 自适应刻度（原方式）
        y_max = max_value * 1.15
    
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111, projection='polar')
    
    # 计算角度：从顶部(90度)开始，顺时针分布
    num_vars = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    # 绘制顺序：基准方法在底层，HOSH在顶层突出显示
    plot_order = [m for m in methods if m != 'HOSH' and m in radar_data]
    if 'HOSH' in radar_data:
        plot_order.append('HOSH')
    
    for m in plot_order:
        values = radar_data[m] + radar_data[m][:1]  # 闭合数据
        
        # 根据方法设置视觉参数
        if m == 'HOSH':
            # HOSH：最突出的视觉效果
            linewidth = 2.0
            markersize = 6.0
            alpha_fill = 0.25
            alpha_line = 1.0
            marker = 'o'
            mfc = colors.get(m, '#D63230')  # 实心标记
            mec = 'black'
            mew = 0.5
            zorder = 100
        else:
            # 其他方法：较淡的视觉效果
            linewidth = 1.4
            markersize = 4.5
            alpha_fill = 0.15
            alpha_line = 0.75
            marker = markers.get(m, 'o')
            mfc = colors.get(m, '#000000')  # 实心标记
            mec = 'black'
            mew = 0.5
            zorder = 50
        
        # 填充区域（半透明）
        ax.fill(angles, values,
                color=colors.get(m, '#000000'),
                alpha=alpha_fill,
                zorder=zorder,
                linewidth=0)
        
        # 边界线+标记点
        ax.plot(angles, values,
                color=colors.get(m, '#000000'),
                linewidth=linewidth,
                linestyle='--',
                alpha=alpha_line,
                marker=marker,
                markersize=markersize,
                markerfacecolor=mfc,
                markeredgecolor=mec,
                markeredgewidth=mew,
                label=m,
                zorder=zorder + 1)
    
    # 雷达轴标签：清晰可读的节点比例
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=9, weight='bold', color='#2C3E50')
    
    # 径向轴：统一刻度 0-1，上限设置为1.03留出标记点空间
    ax.set_ylim(0, y_max * 1.03)
    ax.set_rlabel_position(88)
    
    # 隐藏径向刻度标签
    ax.set_yticklabels([])
    
    # 网格线：细虚线，低对比度
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.30, color='#95A5A6')
    ax.set_axisbelow(True)
    
    # 背景：纯白色
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('#34495E')
    
    # 图例：紧凑单行布局，利用绝对坐标确保完全居中
    legend = ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.1 ),
        frameon=True,
        fancybox=False,
        shadow=False,
        edgecolor='#2C3E50',
        facecolor='white',
        framealpha=1.0,
        ncol=len(plot_order),
        columnspacing=0.5,
        labelspacing=0.2,
        handlelength=1.0,
        handletextpad=0.2,
        borderpad=0.4,
        fontsize=6.5
    )
    legend.get_frame().set_linewidth(0.8)
    
    # 调整布局：为图例和边缘标签预留足够空间
    plt.subplots_adjust(left=0.15, right=0.85, top=0.82, bottom=0.15)
    
    # 保存高分辨率图像
    save_path = os.path.join(output_dir, f"Separation_Radar_{net}.png")
    plt.savefig(save_path, 
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.05,
                facecolor='white',
                edgecolor='none',
                pil_kwargs={'quality': 95, 'optimize': True})
    
    print(f"    [Output] radar chart saved: {save_path}")
    plt.close()

# ==========================================
# 3. 数据导出函数
# ==========================================
def save_separation_results_to_excel(all_results, methods, output_dir):
    """保存传播者分离度结果到Excel"""
    print("\n  [Export] Saving Spreader Separation results to Excel...")
    
    for net, data in all_results.items():
        x_ratios = data['x_ratios']
        separation_results = data['separation_results']
        
        # 构建数据框
        df_data = {'Seed_Ratio_%': x_ratios * 100}
        for m in methods:
            if m in separation_results:
                df_data[f'{m}_AvgDistance'] = separation_results[m]
        
        df = pd.DataFrame(df_data)
        
        # 保存到Excel
        excel_path = os.path.join(output_dir, f"Separation_Data_{net}.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Separation_Results', index=False)
            worksheet = writer.sheets['Separation_Results']
            
            # 调整列宽
            worksheet.column_dimensions['A'].width = 15
            for i, m in enumerate(methods):
                col_letter = chr(ord('B') + i)
                worksheet.column_dimensions[col_letter].width = 16
        
        print(f"    Saved: {excel_path}")

# ==========================================
# 4. 主流程
# ==========================================
def main():
    print("=" * 60)
    print(" Experiment: Spreader Separation Analysis")
    print("=" * 60)
    
    # 输出目录
    output_dir = "results/exp_spreader_separation"
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
    
    # 标记符号 - 易于区分（面积图不使用标记，但保留以便未来扩展）
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
            g = download_and_load_graph(net)
            if g is None or g.number_of_nodes() == 0:
                print(f"  [Skip] Network {net} is empty or failed to load")
                continue
            
            print(f"  Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
            
            x_ratios, separation_results = exp_spreader_separation(methods, g, net)
            plot_separation_results(net, g, x_ratios, separation_results, methods, colors, markers, output_dir)
            
            # 保存结果用于后续导出
            all_results[net] = {
                'x_ratios': x_ratios,
                'separation_results': separation_results
            }
            
        except Exception as e:
            print(f"  [Error] Failed to process {net}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 导出所有结果到Excel
    if all_results:
        save_separation_results_to_excel(all_results, methods, output_dir)
    
    print("\n" + "=" * 60)
    print(" Experiment: Spreader Separation Completed!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()
