"""
实验4: 改进率分析
评估HOSH方法相对其他方法在不同传播阈值下的改进率
横轴: 传播阈值倍数 (1.0 - 1.9, 步长0.1)
纵轴: HOSH相较于其他方法在25%初始感染节点比例下的感染规模改进率
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

def exp_improvement_rate(methods, g, network_name=None):
    """改进率实验"""
    print("  [Exp: Improvement Rate] Running analysis...")

    N = g.number_of_nodes()
    degrees = [d for n, d in g.degree()]
    k_mean = np.mean(degrees)
    k2_mean = np.mean([d ** 2 for d in degrees])

    # 基础传播阈值
    base_beta = k_mean / (k2_mean - k_mean)
    gamma = 0.5

    print(f"    Graph Properties: <k>={k_mean:.2f}, <k^2>={k2_mean:.2f}")
    print(f"    Base beta (threshold): {base_beta:.4f}, gamma={gamma:.2f}")

    # 传播阈值倍数范围: 1.0 到 2.0，步长 0.2
    beta_multipliers = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    
    # 固定初始感染比例为 25%
    seed_ratio = 0.25
    k = int(N * seed_ratio)
    if k == 0: k = 1

    # 统一使用1000次重复实验
    repeat_times = 1000
    print(f"    Simulation Repeats: {repeat_times}")
    print(f"    Fixed seed ratio: {seed_ratio*100:.1f}% ({k} nodes)")

    # 尝试加载预计算排名
    precomputed = load_precomputed_rankings(network_name) if network_name else None

    # 预先计算所有方法的排名节点
    ranked_nodes = {}
    for method_name in methods:
        # 优先使用预计算数据
        if precomputed and method_name in precomputed and precomputed[method_name]:
            print(f"      {method_name}: Using precomputed rankings")
            scores = precomputed[method_name]
        else:
            print(f"      {method_name}: Computing on-the-fly")
            scores = get_node_scores(method_name, g)
        ranked_nodes[method_name] = sorted(scores, key=scores.get, reverse=True)[:k]

    # 对每个传播阈值倍数进行实验
    results = {m: [] for m in methods}
    
    for multiplier in tqdm(beta_multipliers, desc="    Beta multipliers"):
        beta = base_beta * multiplier
        
        # 对每个方法进行模拟
        method_infections = {}
        for method_name in methods:
            seeds = ranked_nodes[method_name]
            
            total_infected = 0
            for _ in range(repeat_times):
                total_infected += run_sir_simulation(g, seeds, beta, gamma)
            
            avg_infected_rate = (total_infected / repeat_times) / N
            method_infections[method_name] = avg_infected_rate
        
        # 保存结果
        for method_name in methods:
            results[method_name].append(method_infections[method_name])

    # 计算改进率: (HOSH - Other) / Other * 100%
    hosh_results = results['HOSH']
    improvement_rates = {}
    infection_scales = {}  # 保存感染规模用于气泡大小
    
    for method_name in methods:
        if method_name == 'HOSH':
            continue
        
        other_results = results[method_name]
        improvements = []
        scales = []
        
        for hosh_val, other_val in zip(hosh_results, other_results):
            if other_val > 0:
                improvement = (hosh_val - other_val) / other_val * 100
            else:
                improvement = 0
            improvements.append(improvement)
            # 保存对应方法的感染规模（百分比）用于气泡大小 - V_prop
            scales.append(other_val * 100)
        
        improvement_rates[method_name] = improvements
        infection_scales[method_name] = scales

    return beta_multipliers, improvement_rates, infection_scales

# ==========================================
# 2. 绘图函数
# ==========================================
def plot_improvement_bubble_heatmap(net, beta_multipliers, improvement_rates, infection_scales, methods, output_dir):
    """绘制气泡热力图"""
    from matplotlib.colors import LinearSegmentedColormap
    
    # 与 SIR 实验完全相同的 figsize
    fig = plt.figure(figsize=(3.5, 2.8))
    
    # 手动布局：主图 + 右侧图例区，确保最终尺寸一致
    # [left, bottom, width, height] 归一化坐标
    ax = fig.add_axes([0.12, 0.15, 0.58, 0.80])
    
    method_names = [m for m in methods if m != 'HOSH' and m in improvement_rates]
    
    all_improvements = []
    all_scales = []
    for m in method_names:
        all_improvements.extend(improvement_rates[m])
        all_scales.extend(infection_scales[m])
    
    vmin = 0
    vmax = max(all_improvements)
    
    # 采用高对比度且跨度更大的渐变色图 (这里使用RdYlBu_r: 蓝->浅黄->红)，
    # 彻底解决单色系导致数值梯度无法区分的问题
    cmap = plt.get_cmap('RdYlBu_r')
    
    scale_min = min(all_scales)
    scale_max = max(all_scales)
    
    def scale_to_size(scale_val):
        if scale_max - scale_min > 0:
            normalized = (scale_val - scale_min) / (scale_max - scale_min)
            return 50 + np.sqrt(normalized) * 350
        else:
            return 200
    
    scatter_ref = None
    for i, method in enumerate(method_names):
        for j, beta_mult in enumerate(beta_multipliers):
            improvement = improvement_rates[method][j]
            infection_scale = infection_scales[method][j]
            size = scale_to_size(infection_scale)
            c_val = max(0, improvement)
            sc = ax.scatter(j, i, s=size, c=[c_val],
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            alpha=0.85, edgecolors='#333333', linewidth=0.5,
                            zorder=5)
            if scatter_ref is None:
                scatter_ref = sc
    
    ax.set_xticks(np.arange(len(beta_multipliers)))
    ax.set_xticklabels([f'{b:.1f}' for b in beta_multipliers], fontsize=9)
    ax.set_yticks(np.arange(len(method_names)))
    ax.set_yticklabels(method_names, fontsize=9)
    
    ax.set_xlabel(r'$\beta/\beta_{th}$', fontsize=11)
    # 删除y轴标签
    ax.set_xlim(-0.5, len(beta_multipliers) - 0.5)
    ax.set_ylim(-0.5, len(method_names) - 0.5)
    
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('#000000')
    ax.tick_params(direction='out', which='major', length=3.0, width=0.7)
    
    # ---- 右侧图例区：调整位置以适应删除y轴标签后的布局 ----
    # 主图扩展到更左侧，图例区域相应调整
    ax.set_position([0.08, 0.15, 0.62, 0.80])
    
    # 气泡大小图例 - 调整位置，增加图例间距
    ax_bubble = fig.add_axes([0.73, 0.53, 0.14, 0.32])
    ax_bubble.set_axis_off()
    
    all_scales_sorted = sorted(all_scales)
    n_s = len(all_scales_sorted)
    legend_scales = [
        all_scales_sorted[max(0, int(n_s * 0.1))],
        all_scales_sorted[int(n_s * 0.5)],
        all_scales_sorted[min(n_s - 1, int(n_s * 0.9))]
    ]
    legend_sizes = [scale_to_size(s) for s in legend_scales]
    legend_labels = [f'{s:.0f}' for s in legend_scales]
    
    ax_bubble.text(0.35, 1.02, r'$F(t_c)$(%)', fontsize=7,
                   ha='center', va='bottom', transform=ax_bubble.transAxes)
    
    # 减少气泡间距，使分布更紧凑，整体向上移动靠近标题
    y_pos = np.linspace(0.85, 0.35, len(legend_scales))
    for idx in range(len(legend_scales)):
        sz = legend_sizes[idx]
        ax_bubble.scatter(0.15, y_pos[idx], s=sz * 0.35,
                          facecolor='white', edgecolor='#333333',
                          linewidth=0.5, transform=ax_bubble.transAxes,
                          clip_on=False, zorder=5)
        # 调整文字位置，避免与气泡重叠
        ax_bubble.text(0.35, y_pos[idx], legend_labels[idx],
                       fontsize=7, va='center', ha='left',
                       transform=ax_bubble.transAxes)
    
    # 颜色条 - 调整位置，增加与气泡图例的间距，减小长度
    ax_cbar = fig.add_axes([0.74, 0.20, 0.025, 0.25])
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation='vertical')
    ax_cbar.set_title(r'$\eta$(%)', fontsize=7, pad=8)
    cbar.set_label('')
    cbar.ax.tick_params(labelsize=7, width=0.4, length=2, pad=1)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color('#000000')
    
    # 保存 - 使用与SIR实验相同的设置以确保清晰度
    save_path = os.path.join(output_dir, f"Improvement_Bubble_{net}.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none', pil_kwargs={'quality': 95})
    
    print(f"    [Output] High-quality bubble heatmap saved: {save_path} (600 dpi)")
    plt.close()

# ==========================================
# 3. 数据导出函数
# ==========================================
def save_improvement_results_to_excel(all_results, methods, output_dir):
    """保存改进率结果到Excel"""
    print("\n  [Export] Saving Improvement Rate results to Excel...")
    
    for net, data in all_results.items():
        beta_multipliers = data['beta_multipliers']
        improvement_rates = data['improvement_rates']
        infection_scales = data['infection_scales']
        
        # 构建数据框 - 改进率
        df_improvement = {'Beta_Multiplier': beta_multipliers}
        for m in methods:
            if m != 'HOSH' and m in improvement_rates:
                df_improvement[f'{m}_Improvement_%'] = improvement_rates[m]
        
        # 构建数据框 - 感染规模
        df_infection = {'Beta_Multiplier': beta_multipliers}
        for m in methods:
            if m != 'HOSH' and m in infection_scales:
                df_infection[f'{m}_Infection_Scale_%'] = infection_scales[m]
        
        df_imp = pd.DataFrame(df_improvement)
        df_inf = pd.DataFrame(df_infection)
        
        # 保存到Excel（两个sheet）
        excel_path = os.path.join(output_dir, f"Improvement_Data_{net}.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_imp.to_excel(writer, sheet_name='Improvement_Rates', index=False)
            df_inf.to_excel(writer, sheet_name='Infection_Scales', index=False)
            
            # 调整列宽
            for sheet_name in ['Improvement_Rates', 'Infection_Scales']:
                worksheet = writer.sheets[sheet_name]
                worksheet.column_dimensions['A'].width = 16
                for i in range(len([m for m in methods if m != 'HOSH'])):
                    col_letter = chr(ord('B') + i)
                    worksheet.column_dimensions[col_letter].width = 18
        
        print(f"    Saved: {excel_path}")

# ==========================================
# 4. 主流程
# ==========================================
def main():
    print("=" * 60)
    print(" Experiment: Improvement Rate Analysis")
    print("=" * 60)
    
    set_seed(42)
    
    # 输出目录
    output_dir = "results/exp_improvement_rate"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用统一的网络列表
    networks = get_network_list()

    # 统一的方法列表
    methods = ['HOSH', 'ISH', 'DC', 'BC', 'CC', 'K-Shell', 'SH', 'CI', 'SNC']
    
    # 存储所有网络的结果
    all_results = {}

    # 基于实验1配色但降低饱和度的柱状图配色方案
    colors = {
        'ISH': '#E8AE7C',       # 浅橙色
        'DC': '#E8C794',        # 浅金色
        'BC': '#8BB8D8',        # 浅蓝色
        'CC': '#7A8FBF',        # 浅靛蓝
        'K-Shell': '#B884B8',   # 浅紫色
        'SH': '#E8A1B8',        # 浅粉色
        'CI': '#A1887F',        # 浅棕色
        'SNC': '#80CBC4'        # 浅蓝绿色
    }
    
    markers = {
        'HOSH': 'o',      # 圆形
        'ISH': 's',       # 正方形
        'DC': '^',        # 上三角
        'BC': 'v',        # 下三角
        'CC': 'D',        # 菱形
        'K-Shell': 'p',   # 五边形
        'SH': '*',        # 星形
        'CI': 'h',        # 平顶六边形
        'SNC': 'H'        # 尖顶六边形
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
            
            beta_multipliers, improvement_rates, infection_scales = exp_improvement_rate(methods, g, net)
            plot_improvement_bubble_heatmap(net, beta_multipliers, improvement_rates, infection_scales, methods, output_dir)
            
            # 保存结果用于后续导出
            all_results[net] = {
                'beta_multipliers': beta_multipliers,
                'improvement_rates': improvement_rates,
                'infection_scales': infection_scales
            }
            
        except Exception as e:
            print(f"  [Error] Failed to process {net}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
    # 导出所有结果到Excel
    if all_results:
        save_improvement_results_to_excel(all_results, methods, output_dir)
    
    print("\n" + "=" * 60)
    print(" Experiment: Improvement Rate Completed!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()
