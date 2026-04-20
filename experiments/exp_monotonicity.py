"""
实验2: 排名单调性实验
评估不同方法的节点排名区分能力，输出到Excel表格
"""
import networkx as nx
import numpy as np
import random
import os
import pandas as pd

# 导入方法实现
from hosh_methods import get_node_scores
# 导入网络加载模块
from network_loader import download_and_load_graph, get_network_list
# 导入预计算排名模块
from precompute_rankings import load_precomputed_rankings

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    print(f"[Init] Global seed set to: {seed}")

# 网络加载功能已移至 network_loader.py
# 直接使用: from network_loader import download_and_load_graph

# ==========================================
# 单调性分析
# ==========================================
def exp_monotonicity(methods, g, network_name=None):
    """
    计算 M(R) 指标，衡量排名的分辨能力
    M(R) = (1 - sum(nr * (nr - 1)) / (N * (N - 1)))^2
    """
    print("  [Exp: Monotonicity] Running M(R) analysis...")
    results = {}
    N = g.number_of_nodes()

    if N <= 1:
        return {m: 1.0 for m in methods}

    # 尝试加载预计算排名
    precomputed = load_precomputed_rankings(network_name) if network_name else None

    for m in methods:
        # 优先使用预计算数据
        if precomputed and m in precomputed and precomputed[m]:
            print(f"    {m}: Using precomputed rankings")
            scores = precomputed[m]
        else:
            print(f"    {m}: Computing on-the-fly")
            scores = get_node_scores(m, g)
        values = list(scores.values())

        _, counts = np.unique(values, return_counts=True)
        same_rank_pairs = np.sum(counts * (counts - 1))
        total_pairs = N * (N - 1)
        mr = (1.0 - same_rank_pairs / total_pairs) ** 2
        results[m] = mr

    return results

def save_monotonicity_to_excel(all_results, methods, output_dir):
    """将单调性结果保存到Excel文件"""
    
    print("\n  [Export] Saving Monotonicity results to Excel...")
    
    rows = []
    for net, data in all_results.items():
        row = {
            'Network': net.upper(),
            'Nodes': data['N'],
            'Edges': data['E']
        }
        for m in methods:
            row[m] = data.get(m, 0.0)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    columns = ['Network', 'Nodes', 'Edges'] + methods
    df = df[columns]
    
    # 计算每个方法的平均值和方差
    stats_rows = []
    
    # 平均值行
    mean_row = {'Network': 'MEAN', 'Nodes': '', 'Edges': ''}
    for m in methods:
        mean_row[m] = df[m].mean()
    stats_rows.append(mean_row)
    
    # 方差行
    var_row = {'Network': 'VARIANCE', 'Nodes': '', 'Edges': ''}
    for m in methods:
        var_row[m] = df[m].var()
    stats_rows.append(var_row)
    
    # 标准差行
    std_row = {'Network': 'STD', 'Nodes': '', 'Edges': ''}
    for m in methods:
        std_row[m] = df[m].std()
    stats_rows.append(std_row)
    
    # 添加统计行到数据框
    stats_df = pd.DataFrame(stats_rows)
    df = pd.concat([df, stats_df], ignore_index=True)
    
    excel_path = os.path.join(output_dir, "Monotonicity_Results.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Monotonicity M(R)', index=False)
        worksheet = writer.sheets['Monotonicity M(R)']
        
        worksheet.column_dimensions['A'].width = 15
        worksheet.column_dimensions['B'].width = 10
        worksheet.column_dimensions['C'].width = 10
        for i, m in enumerate(methods):
            col_letter = chr(ord('D') + i)
            worksheet.column_dimensions[col_letter].width = 15
        
        for row in range(2, len(df) + 2):
            for col in range(4, 4 + len(methods)):
                cell = worksheet.cell(row=row, column=col)
                if isinstance(cell.value, (int, float)):
                    cell.number_format = '0.0000'
    
    print(f"    [Output] Excel saved: {excel_path}")
    
    csv_path = os.path.join(output_dir, "Monotonicity_Results.csv")
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"    [Output] CSV backup saved: {csv_path}")
    
    print("\n  [Summary] Monotonicity M(R) Results:")
    print("  " + "-" * 90)
    header = f"  {'Network':<12} {'N':<8} {'E':<8}"
    for m in methods:
        header += f" {m:<12}"
    print(header)
    print("  " + "-" * 90)
    
    for _, row in df.iterrows():
        net_name = str(row['Network'])
        nodes_val = str(row['Nodes']) if row['Nodes'] != '' else ''
        edges_val = str(row['Edges']) if row['Edges'] != '' else ''
        
        line = f"  {net_name:<12} {nodes_val:<8} {edges_val:<8}"
        for m in methods:
            line += f" {row[m]:<12.4f}"
        print(line)
        
        # 在统计行前添加分隔线
        if net_name not in ['MEAN', 'VARIANCE', 'STD'] and _ == len(df) - 4:
            print("  " + "-" * 90)
    
    print("  " + "-" * 90)

# ==========================================
# 主流程
# ==========================================
def main():
    print("=" * 60)
    print(" Experiment: Monotonicity Analysis")
    print("=" * 60)
    
    set_seed(42)
    
    # 输出目录
    output_dir = "results/exp_monotonicity"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用统一的网络列表
    networks = get_network_list()

    # 统一的方法列表
    methods = ['HOSH', 'ISH', 'DC', 'BC', 'CC', 'K-Shell', 'SH', 'CI', 'SNC']

    all_monotonicity_results = {}

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
            
            mr_results = exp_monotonicity(methods, g, net)
            
            all_monotonicity_results[net] = {
                'N': g.number_of_nodes(),
                'E': g.number_of_edges(),
                **mr_results
            }
            
            print("  [Results] Monotonicity M(R):")
            for m, val in mr_results.items():
                print(f"    {m:15s}: {val:.4f}")
        
        except Exception as e:
            print(f"  [Error] Failed to process {net}: {e}")
            import traceback
            traceback.print_exc()
            continue

    save_monotonicity_to_excel(all_monotonicity_results, methods, output_dir)
    
    print("\n" + "=" * 60)
    print(" Experiment: Monotonicity Completed!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()