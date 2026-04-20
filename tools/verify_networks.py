"""
网络数据验证脚本
用于检查12个真实网络的下载、加载和LCC处理是否正确
"""
import networkx as nx
from network_loader import download_and_load_graph

def verify_networks():
    """验证所有网络数据集"""
    
    networks = [
        ('polblogs', 1222),
        ('hamster', 2426),
        ('lesmis', 77),
        ('adjnoun', 112),
        ('jazz', 198),
        ('usair', 332),
        ('email', 1133),

        ('pgp', 10680),

        ('facebook',4091)
    ]
    
    print("=" * 80)
    print("网络数据验证报告")
    print("=" * 80)
    print(f"{'网络名称':<15} {'预期节点数':<12} {'实际节点数':<12} {'边数':<10} {'平均度':<10} {'连通性'}")
    print("-" * 80)
    
    results = []
    
    for net_name, expected_nodes in networks:
        try:
            g = download_and_load_graph(net_name)
            
            n = g.number_of_nodes()
            m = g.number_of_edges()
            avg_degree = 2 * m / n if n > 0 else 0
            is_connected = nx.is_connected(g)
            
            # 检查节点数是否接近预期(允许±10%误差,因为LCC提取)
            node_diff_pct = abs(n - expected_nodes) / expected_nodes * 100
            status = "✓" if node_diff_pct < 10 else "⚠"
            
            print(f"{net_name:<15} {expected_nodes:<12} {n:<12} {m:<10} {avg_degree:<10.2f} {is_connected}")
            
            results.append({
                'name': net_name,
                'expected': expected_nodes,
                'actual': n,
                'edges': m,
                'connected': is_connected,
                'status': status
            })
            
        except Exception as e:
            print(f"{net_name:<15} {expected_nodes:<12} {'ERROR':<12} {str(e)[:40]}")
            results.append({
                'name': net_name,
                'expected': expected_nodes,
                'actual': None,
                'edges': None,
                'connected': False,
                'status': "✗"
            })
    
    print("-" * 80)
    
    # 统计
    success_count = sum(1 for r in results if r['actual'] is not None)
    connected_count = sum(1 for r in results if r['connected'])
    
    print(f"\n总结:")
    print(f"  成功加载: {success_count}/{len(networks)}")
    print(f"  连通网络: {connected_count}/{len(networks)}")
    print(f"  总节点数: {sum(r['actual'] for r in results if r['actual'] is not None):,}")
    print(f"  总边数: {sum(r['edges'] for r in results if r['edges'] is not None):,}")
    
    print("\n注意事项:")
    print("  - 实际节点数可能略少于预期,因为LCC提取会移除孤立节点")
    print("  - 所有网络应该都是连通的(已提取LCC)")
    print("  - 如果下载失败,请检查网络连接或手动下载数据集")
    
    return results

if __name__ == "__main__":
    verify_networks()
