"""
网络数据加载模块
提供统一的网络数据下载和加载功能
"""
import networkx as nx
import os
import tarfile
import gzip
import zipfile
import urllib.request
from scipy.io import mmread


# 数据存储目录
DATA_DIR = "networks_data"

# 网络数据配置
NETWORK_URLS = {
    'karate': "https://sparse.tamu.edu/MM/Newman/karate.tar.gz",
    'dolphins': "https://sparse.tamu.edu/MM/Newman/dolphins.tar.gz",
    'jazz': "https://sparse.tamu.edu/MM/Arenas/jazz.tar.gz",
    'usair': "https://sparse.tamu.edu/MM/Pajek/USAir97.tar.gz",
    'celegans': "https://sparse.tamu.edu/MM/Arenas/celegans_metabolic.tar.gz",
    'netscience': "https://sparse.tamu.edu/MM/Newman/netscience.tar.gz",
    'email': "https://sparse.tamu.edu/MM/Arenas/email.tar.gz",
    'lesmis': "https://sparse.tamu.edu/MM/Newman/lesmis.tar.gz",
    'adjnoun': "https://sparse.tamu.edu/MM/Newman/adjnoun.tar.gz",
    'polblogs': "https://sparse.tamu.edu/MM/Newman/polblogs.tar.gz",
    'hamster': "http://nrvis.com/download/data/soc/soc-hamsterster.zip",
    'power': "https://sparse.tamu.edu/MM/Pajek/power.tar.gz",
    'pgp': "https://sparse.tamu.edu/MM/Arenas/PGPgiantcompo.tar.gz",
    'lastfm_asia': "local",  # 本地CSV文件
    'web-spam': "https://sparse.tamu.edu/MM/Pajek/web-spam.tar.gz",
    'infect': "local",  # 本地MTX文件 (ia-infect-dublin)
    'chesapeake': "local"  # 本地MTX文件
}

NETWORK_PATHS = {
    'karate': "karate/karate.mtx",
    'dolphins': "dolphins/dolphins.mtx",
    'jazz': "jazz/jazz.mtx",
    'usair': "USAir97/USAir97.mtx",
    'celegans': "celegans_metabolic/celegans_metabolic.mtx",
    'netscience': "netscience/netscience.mtx",
    'email': "email/email.mtx",
    'lesmis': "lesmis/lesmis.mtx",
    'adjnoun': "adjnoun/adjnoun.mtx",
    'polblogs': "polblogs/polblogs.mtx",
    'hamster': "soc-hamsterster.edges",
    'power': "power/power.mtx",
    'pgp': "PGPgiantcompo/PGPgiantcompo.mtx",
    'lastfm_asia': "lastfm_asia_edges.csv",
    'web-spam': "web-spam/web-spam.mtx",
    'infect': "ia-infect-dublin.mtx",
    'chesapeake': "road-chesapeake.mtx"
}
# 真实网络列表（按节点数排序）
NETWORK_LIST = [
   # 'chesapeake',  # 39节点 (road-chesapeake)
    'lesmis',      # 77节点
    #'adjnoun',     # 112节点
    #'jazz',        # 198节点
    #'usair',       # 332节点
   # 'netscience',  # 379节点
    #'infect',  # 410节点 (ia-infect-dublin)
    #'celegans',    # 453节点
    #'email',       # 1133节点
    #'polblogs',    # 1490节点
    'hamster',     # 2426节点
    'power',       # 4941节点

   # 'web-spam',    # 4767节点
   # 'lastfm_asia', # 7624节点
   # 'pgp'          # 10680节点
]


def download_and_load_graph(network_name: str, verbose: bool = True) -> nx.Graph:
    """
    下载并加载真实网络数据集 (SuiteSparse Matrix Collection)
    
    参数:
        network_name: 网络名称 (如 'lesmis', 'jazz' 等)
        verbose: 是否显示详细信息
        
    返回:
        NetworkX Graph 对象（无向、已提取LCC、节点重标号）
    """
    if network_name not in NETWORK_URLS:
        if verbose:
            print(f"[Error] Network '{network_name}' not defined in URL list.")
        return None

    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    
    url = NETWORK_URLS[network_name]
    extract_file = os.path.join(DATA_DIR, NETWORK_PATHS.get(network_name, f"{network_name}/{network_name}.mtx"))
    
    # 特殊处理：本地文件
    if url == "local":
        if not os.path.exists(extract_file):
            if verbose:
                print(f"  [Error] Local file not found: {extract_file}")
                print(f"  Please place the file in {DATA_DIR}/ directory")
            return None
    else:
        # 根据URL类型确定文件名
        if url.endswith('.tar.gz'):
            archive_name = os.path.join(DATA_DIR, f"{network_name}.tar.gz")
        elif url.endswith('.txt.gz'):
            archive_name = os.path.join(DATA_DIR, f"{network_name}.txt.gz")
        elif url.endswith('.zip'):
            archive_name = os.path.join(DATA_DIR, f"{network_name}.zip")
        else:
            archive_name = os.path.join(DATA_DIR, f"{network_name}.dat")
        
        # 下载逻辑
        if not os.path.exists(extract_file):
            if not os.path.exists(archive_name):
                if verbose:
                    print(f"  [Download] Downloading {network_name}...")
                try:
                    req = urllib.request.Request(url, 
                                                headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req) as r, open(archive_name, 'wb') as f:
                        import shutil
                        shutil.copyfileobj(r, f)
                except Exception as e:
                    if verbose:
                        print(f"  [Error] Download failed: {e}")
                    return None

            # 解压到数据目录
            try:
                if url.endswith('.tar.gz'):
                    # 解压tar.gz文件
                    with tarfile.open(archive_name, "r:gz") as tar:
                        tar.extractall(path=DATA_DIR, filter='data')
                elif url.endswith('.txt.gz'):
                    # 解压.txt.gz文件
                    with gzip.open(archive_name, 'rb') as f_in:
                        with open(extract_file, 'wb') as f_out:
                            import shutil
                            shutil.copyfileobj(f_in, f_out)
                elif url.endswith('.zip'):
                    # 解压.zip文件
                    with zipfile.ZipFile(archive_name, 'r') as zip_ref:
                        zip_ref.extractall(DATA_DIR)
            except Exception as e:
                if verbose:
                    print(f"  [Error] Extraction failed: {e}")

    # 加载网络
    try:
        if extract_file.endswith('.mtx'):
            # MTX格式 (Matrix Market)
            adj = mmread(extract_file).asfptype()
            g = nx.from_scipy_sparse_array(adj)
        elif extract_file.endswith('.csv'):
            # CSV格式 (带header的边列表)
            import pandas as pd
            df = pd.read_csv(extract_file)
            # 假设CSV有两列：node_1, node_2 或类似的列名
            if len(df.columns) >= 2:
                g = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1])
            else:
                raise ValueError("CSV file must have at least 2 columns")
        elif extract_file.endswith('.edges'):
            # .edges格式 (Network Repository格式，通常第一行是注释)
            g = nx.read_edgelist(extract_file, nodetype=int, comments='%')
        else:
            # 文本边列表格式 (如Facebook数据集)
            g = nx.read_edgelist(extract_file, nodetype=int, comments='#')
    except Exception as e:
        # 尝试直接读取 EdgeList (兼容性)
        try:
            g = nx.read_edgelist(extract_file)
        except:
            if verbose:
                print(f"  [Error] Failed to load {network_name}: {e}")
            return None

    # 预处理：无向化、去自环、提取LCC、重标号
    original_nodes = g.number_of_nodes()
    
    g = g.to_undirected()
    g.remove_edges_from(nx.selfloop_edges(g))
    
    # 提取最大连通分量(LCC)
    if not nx.is_connected(g):
        components = list(nx.connected_components(g))
        lcc = max(components, key=len)
        g = g.subgraph(lcc).copy()
        if verbose:
            print(f"  [LCC] Extracted largest connected component: "
                  f"{len(lcc)}/{original_nodes} nodes ({len(lcc)/original_nodes*100:.1f}%)")
    
    g = nx.convert_node_labels_to_integers(g, first_label=0)

    if verbose:
        print(f"  [Load] {network_name}: Nodes={g.number_of_nodes()}, Edges={g.number_of_edges()}")
    
    return g


def get_network_list():
    """
    获取所有可用的网络列表
    
    返回:
        网络名称列表
    """
    return NETWORK_LIST.copy()


def get_network_info(network_name: str) -> dict:
    """
    获取网络的基本信息（不加载网络）
    
    参数:
        network_name: 网络名称
        
    返回:
        包含网络信息的字典
    """
    if network_name not in NETWORK_URLS:
        return None
    
    return {
        'name': network_name,
        'url': NETWORK_URLS[network_name],
        'path': NETWORK_PATHS[network_name]
    }


if __name__ == "__main__":
    """测试模块功能"""
    print("=" * 60)
    print("网络数据加载模块测试")
    print("=" * 60)
    
    # 测试加载一个网络
    test_network = 'grqc'
    print(f"\n测试加载网络: {test_network}")
    g = download_and_load_graph(test_network)
    
    if g:
        print(f"\n✓ 成功加载网络")
        print(f"  节点数: {g.number_of_nodes()}")
        print(f"  边数: {g.number_of_edges()}")
        print(f"  平均度: {2 * g.number_of_edges() / g.number_of_nodes():.2f}")
    
    # 显示所有可用网络
    print(f"\n可用网络列表 ({len(NETWORK_LIST)}个):")
    for i, net in enumerate(NETWORK_LIST, 1):
        print(f"  {i:2d}. {net}")
    
    print("=" * 60)
