import networkx as nx
import numpy as np


def calculate_hosh(g, xi=0.001):
    """
    High-Order Structural Hole (HOSH) 指标实现
    基于极大团的高阶结构洞方法
    
    核心公式：
    步骤1: 基础拆分依赖 p_{iα} = 1 / |M(v_i)|
           总连接能力 k_{i,α}^{total} = k_i^{ext} + ln(1 + Σ_{v_j ∈ C_α \ i} k_j^{ext})
           修正依赖 p_{iα}^* = p_{iα} × (1 - k_{i,α}^{total} / max_{v_m ∈ C_α}(k_{m,α}^{total}) + ξ)
    步骤2: 团间冗余 O_{βα} = |(C_β ∩ C_α) \ v_i| / (|C_β| - 1)
    步骤3: 高阶约束 c_{iα} = (p_{iα}^* + Σ_{C_β ∈ M(v_i), β≠α} p_{iβ}^* · O_{βα})^2
           总约束 C_i = Σ_{C_α ∈ M(v_i)} c_{iα}
           最终分数 score = 1 - C_i
    
    参数:
    g: networkx.Graph (无向图)
    xi: 防止分母为 0 的小量 (默认 0.001)
    
    返回:
    dict: {node_id: hosh_score} (分数越高代表结构洞能力越强，即约束越小)
    """
    # 获取所有极大团 (maximal cliques)
    cliques = list(nx.find_cliques(g))

    # 构建映射: 节点 -> 包含该节点的所有极大团索引 M(v_i)
    node_cliques_map = {n: [] for n in g.nodes()}
    for idx, c in enumerate(cliques):
        for n in c:
            node_cliques_map[n].append(idx)

    # 预计算
    clique_sizes = [len(c) for c in cliques]
    clique_sets = [set(c) for c in cliques]
    global_degrees = dict(g.degree())

    # =========================================================
    # 步骤 1: 计算基础拆分依赖、连接能力和修正依赖
    # =========================================================

    # 1.1 计算每个团内每个节点的总连接能力 k_total
    # 公式：k_{i,α}^{total} = k_i^{ext} + ln(1 + Σ_{v_j ∈ C_α \ i} k_j^{ext})
    clique_node_k_totals = []

    for idx, c_nodes in enumerate(cliques):
        c_size = clique_sizes[idx]
        current_clique_k_data = {}

        # 计算团内所有节点的外部连接能力 k_ext
        node_k_ext_map = {}
        total_k_ext_in_clique = 0.0

        for node in c_nodes:
            # 外部连接 = 总度数 - 团内连接数
            k_ext = max(0, global_degrees[node] - (c_size - 1))
            node_k_ext_map[node] = k_ext
            total_k_ext_in_clique += k_ext

        # 计算总连接能力
        for node in c_nodes:
            k_i_ext = node_k_ext_map[node]

            if c_size > 1:
                # 获取团内其他节点的外部连接总和：Σ_{v_j ∈ C_α \ i} k_j^{ext}
                sum_neighbors_ext = total_k_ext_in_clique - k_i_ext

                # 对数衰减：ln(1 + sum)
                log_sum_ext = np.log1p(sum_neighbors_ext)  # log1p(x) = log(1+x)，数值更稳定

                implicit_capability = log_sum_ext
            else:
                implicit_capability = 0.0

            k_total = k_i_ext + implicit_capability
            current_clique_k_data[node] = k_total

        clique_node_k_totals.append(current_clique_k_data)

    # 1.2 计算修正后的依赖系数 p*_{iα}
    # 公式：p_{iα}^* = p_{iα} × (1 - k_{i,α}^{total} / (max_{v_m ∈ C_α}(k_{m,α}^{total}) + ξ))
    node_p_stars = {n: {} for n in g.nodes()}

    for v in g.nodes():
        my_indices = node_cliques_map[v]
        m_v = len(my_indices)

        if m_v == 0:
            continue

        # 基础拆分依赖：p_{iα} = 1 / |M(v_i)|
        p_base = 1.0 / m_v

        for alpha in my_indices:
            k_totals_in_alpha = clique_node_k_totals[alpha]
            max_k_total_alpha = max(k_totals_in_alpha.values())
            k_total_i_alpha = k_totals_in_alpha[v]

            # 修正依赖：p_{iα}^* = p_{iα} × (1 - k_{i,α}^{total} / (max_{v_m ∈ C_α}(k_{m,α}^{total}) + ξ))
            autonomy_factor = k_total_i_alpha / (max_k_total_alpha + xi)
            p_star = p_base * (1.0 - autonomy_factor)

            node_p_stars[v][alpha] = p_star

    # =========================================================
    # 步骤 2: 团间冗余
    # 公式：O_{βα} = |(C_β ∩ C_α) \ v_i| / (|C_β| - 1)
    # =========================================================

    # =========================================================
    # 步骤 3: 高阶约束系数
    # 公式：c_{iα} = (p_{iα}^* + Σ_{C_β ∈ M(v_i), β≠α} p_{iβ}^* · O_{βα})^2
    # =========================================================

    scores = {}

    for v in g.nodes():
        my_indices = node_cliques_map[v]
        if not my_indices:
            scores[v] = 0.0
            continue

        total_constraint_i = 0.0

        for alpha in my_indices:
            p_star_alpha = node_p_stars[v][alpha]
            
            # 计算间接约束：Σ_{C_β ∈ M(v_i), β≠α} p_{iβ}^* · O_{βα}
            indirect_constraint_sum = 0.0

            for beta in my_indices:
                if beta == alpha:
                    continue

                p_star_beta = node_p_stars[v][beta]

                # 团间冗余：O_{βα} = |(C_β ∩ C_α) \ v_i| / (|C_β| - 1)
                intersection_size = len(clique_sets[beta].intersection(clique_sets[alpha]))
                numerator = max(0, intersection_size - 1)  # 减去节点 v_i 自身
                denominator = clique_sizes[beta] - 1

                if denominator > 0:
                    o_beta_alpha = numerator / denominator
                else:
                    o_beta_alpha = 0.0

                indirect_constraint_sum += p_star_beta * o_beta_alpha

            # 高阶约束：c_{iα} = (p_{iα}^* + Σ_{C_β ∈ M(v_i), β≠α} p_{iβ}^* · O_{βα})^2
            c_i_alpha = (p_star_alpha + indirect_constraint_sum) ** 2
            total_constraint_i += c_i_alpha

        # 最终分数：score = 1 - C_i
        scores[v] = 1.0 - total_constraint_i

    return scores


def calculate_ish(g):
    """
    Improved Structural Hole (ISH) 指标实现
    基于相对重要性的改进结构洞方法
    
    参数:
    g: networkx.Graph (无向图)
    
    返回:
    dict: {node_id: ish_score} (分数越高代表结构洞能力越强)
    """
    scores = {}
    
    # 计算所有节点的度
    degrees = dict(g.degree())
    
    for i in g.nodes():
        k_i = degrees[i]
        
        if k_i == 0:
            scores[i] = 0.0
            continue
        
        # 计算边权重 w_ij = k(i) + k(j)
        edge_weights = {}
        for j in g.neighbors(i):
            edge_weights[j] = k_i + degrees[j]
        
        # 计算节点权重 w(i) = Σ w_ij
        w_i = sum(edge_weights.values())
        
        if w_i == 0:
            scores[i] = 0.0
            continue
        
        # 计算相对重要性 p(i,j) = w_ij / w(i)
        relative_importance = {}
        for j in g.neighbors(i):
            relative_importance[j] = edge_weights[j] / w_i
        
        # 计算邻居之间的相对重要性归一化
        sum_p_ij = sum(relative_importance.values())
        if sum_p_ij > 0:
            for j in relative_importance:
                relative_importance[j] = relative_importance[j] / sum_p_ij
        
        # 计算约束 C(i) = Σ_j (p(i,j) + Σ_q p(i,q) * p(q,j))^2
        constraint = 0.0
        
        for j in g.neighbors(i):
            p_ij = relative_importance[j]
            
            # 计算间接约束: Σ_q p(i,q) * p(q,j)
            indirect = 0.0
            for q in g.neighbors(i):
                if q == j:
                    continue
                
                p_iq = relative_importance[q]
                
                # 检查 q 和 j 是否相连
                if g.has_edge(q, j):
                    # 计算 p(q,j): q对j的相对重要性
                    k_q = degrees[q]
                    w_q = sum(k_q + degrees[neighbor] for neighbor in g.neighbors(q))
                    
                    if w_q > 0:
                        w_qj = k_q + degrees[j]
                        p_qj = w_qj / w_q
                        
                        # 归一化
                        sum_p_qn = sum((k_q + degrees[n]) / w_q for n in g.neighbors(q))
                        if sum_p_qn > 0:
                            p_qj = p_qj / sum_p_qn
                        
                        indirect += p_iq * p_qj
            
            # 计算该邻居的约束贡献
            constraint += (p_ij + indirect) ** 2
        
        # ISH分数 = 1 - Constraint (约束越小，结构洞能力越强)
        scores[i] = 1.0 - constraint
    
    return scores


def calculate_sh(g):
    """
    Traditional Structural Hole (SH) 指标实现 - Burt's Constraint
    基于Burt (1992) 的经典结构洞理论
    
    核心公式:
    约束系数 C_i = Σ_j (p_ij + Σ_q p_iq * p_qj)^2
    其中 p_ij = (1 + Σ_k∈Ni∩Nj w_ik) / (Σ_k∈Ni w_ik)
    
    参数:
    g: networkx.Graph (无向图)
    
    返回:
    dict: {node_id: sh_score} (分数越高代表结构洞能力越强，即约束越小)
    """
    scores = {}
    
    # 计算所有节点的度
    degrees = dict(g.degree())
    
    for i in g.nodes():
        k_i = degrees[i]
        
        if k_i == 0:
            scores[i] = 0.0
            continue
        
        # 获取邻居集合
        neighbors_i = set(g.neighbors(i))
        
        # 计算比例投资 p_ij = 1 / k_i (简化版本，假设边权重相等)
        p_ij_base = 1.0 / k_i
        
        # 计算约束系数
        constraint = 0.0
        
        for j in neighbors_i:
            # 直接约束: p_ij
            p_ij = p_ij_base
            
            # 间接约束: Σ_q p_iq * p_qj
            indirect = 0.0
            
            for q in neighbors_i:
                if q == j:
                    continue
                
                # 检查 q 和 j 是否相连
                if g.has_edge(q, j):
                    k_q = degrees[q]
                    if k_q > 0:
                        p_iq = p_ij_base
                        p_qj = 1.0 / k_q
                        indirect += p_iq * p_qj
            
            # 约束贡献: (p_ij + Σ_q p_iq * p_qj)^2
            constraint += (p_ij + indirect) ** 2
        
        # SH分数 = 1 - Constraint (约束越小，结构洞能力越强)
        scores[i] = 1.0 - constraint
    
    return scores


def calculate_ci(g, radius=2):
    """
    Collective Influence (CI) 指标实现
    (k_i - 1) * sum_{j in Ball(i, radius)} (k_j - 1)
    """
    scores = {}
    degrees = dict(g.degree())
    
    for i in g.nodes():
        if degrees[i] <= 1:
            scores[i] = 0.0
            continue
            
        visited = {i}
        current_level = [i]
        
        # BFS 到达第 radius 层
        for _ in range(radius):
            next_level = []
            for node in current_level:
                for neighbor in g.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
            current_level = next_level
            
        # 计算第 radius 层上节点的 (k_j - 1) 之和
        sum_kj_minus_1 = sum(max(0, degrees[j] - 1) for j in current_level)
        scores[i] = (degrees[i] - 1) * sum_kj_minus_1
        
    return scores


def calculate_snc(g):
    """
    Structural Neighborhood Centrality (SNC)
    根据论文流程:
    1. mass_i = d_i * e^{C_i}
    2. omega_i = e^{-SH_i} / 2 (SH_i 为 Burt's constraint)
    3. nc_i = sum_{j in N1(i) U N2(i) U {i}} mass_j * mass_i * omega_j
    4. snc_i = nc_i * omega_i
    """
    scores = {}
    degrees = dict(g.degree())
    clustering = nx.clustering(g)
    
    # 获取 Burt's constraint (由于当前 calculate_sh 返回 1 - constraint，需要还原)
    sh_scores = calculate_sh(g)
    
    mass = {}
    omega = {}
    
    for i in g.nodes():
        constraint_i = 1.0 - sh_scores.get(i, 0.0)
        mass[i] = degrees[i] * np.exp(clustering.get(i, 0.0))
        omega[i] = np.exp(-constraint_i) / 2.0
        
    for i in g.nodes():
        # 获取 1阶、2阶邻居以及节点本身 (论文中提及 include the node itself)
        neighbors_1 = set(g.neighbors(i))
        neighbors_2 = set()
        for n1 in neighbors_1:
            neighbors_2.update(g.neighbors(n1))
            
        neighborhood = neighbors_1.union(neighbors_2)
        neighborhood.add(i)
        
        nc_i = sum(mass[j] * mass[i] * omega[j] for j in neighborhood)
        scores[i] = nc_i * omega[i]
        
    return scores


def get_node_scores(method, g):
    """
    统一算法调用接口
    """
    if method == 'HOSH':
        return calculate_hosh(g)
    
    elif method == 'ISH':  # Improved Structural Hole
        return calculate_ish(g)

    elif method == 'DC':  # Degree Centrality
        # 返回归一化的度中心性
        return nx.degree_centrality(g)

    elif method == 'BC':  # Betweenness Centrality
        return nx.betweenness_centrality(g)

    elif method == 'CC':  # Closeness Centrality
        return nx.closeness_centrality(g)

    elif method == 'K-Shell':
        # K-Shell分解（Core number）
        return dict(nx.core_number(g))

    elif method == 'SH':
        # 传统的 Burt's Constraint (Structural Hole)
        # 使用自己实现的版本，确保公平比较
        return calculate_sh(g)

    elif method == 'CI':
        return calculate_ci(g)

    elif method == 'SNC':
        return calculate_snc(g)

    else:
        raise ValueError(f"Unknown Method: {method}")