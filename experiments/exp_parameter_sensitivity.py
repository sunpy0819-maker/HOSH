import os
import pandas as pd
from scipy.stats import kendalltau
from hosh_methods import calculate_hosh
from network_loader import download_and_load_graph, get_network_list

def run_sensitivity_experiment():
    print("=" * 60)
    print(" Experiment: Parameter Sensitivity (Kendall's Tau)")
    print("=" * 60)
    
    # 输出目录
    output_dir = "results/exp_parameter_sensitivity"
    os.makedirs(output_dir, exist_ok=True)
    
    # 待使用的网络及 xi 参数列表
    networks = get_network_list()
    xi_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    baseline_xi = 1e-3
    
    results = []
    
    for net in networks:
        print(f"\nProcessing network: {net}")
        try:
            g = download_and_load_graph(net)
            if g is None or g.number_of_nodes() == 0:
                print(f"  [Skip] Network {net} is empty or failed to load")
                continue
                
            nodes = list(g.nodes())
            
            # 计算基准排名 (xi = 10^-3)
            print(f"  Computing baseline (xi = {baseline_xi})...")
            baseline_scores = calculate_hosh(g, xi=baseline_xi)
            # 按照相同的节点顺序提取分数
            baseline_values = [baseline_scores[n] for n in nodes]
            
            # 记录当前网络的 Kendall's Tau
            tau_results = {'Network': net}
            
            for xi in xi_values:
                print(f"  Computing xi = {xi}...")
                if xi == baseline_xi:
                    tau_results[f"xi={xi}"] = 1.0
                else:
                    scores = calculate_hosh(g, xi=xi)
                    values = [scores[n] for n in nodes]
                    
                    # 也可以根据分数对节点排序再输入排名计算，
                    # 考虑到 kendalltau 函数可以通过两组变量值自动处理顺序，这里直接送入节点的分数
                    tau, p_value = kendalltau(baseline_values, values)
                    tau_results[f"xi={xi}"] = tau
                    
            results.append(tau_results)
            
        except Exception as e:
            print(f"  [Error] Failed to process {net}: {e}")
            continue
            
    # 转换为 DataFrame 生成表格
    df = pd.DataFrame(results)
    print("\n=== Sensitivity Results (Kendall's Tau) ===")
    print(df.to_string(index=False))
    
    # 导出到 Excel 和 CSV
    excel_path = os.path.join(output_dir, "parameter_sensitivity_results.xlsx")
    csv_path = os.path.join(output_dir, "parameter_sensitivity_results.csv")
    
    df.to_excel(excel_path, index=False)
    df.to_csv(csv_path, index=False)
    
    print(f"\n[Export] Results saved to:")
    print(f"  - {excel_path}")
    print(f"  - {csv_path}")

if __name__ == "__main__":
    run_sensitivity_experiment()
