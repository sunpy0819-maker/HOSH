HOSH: 基于极大团的高阶结构洞关键节点识别方法 (Higher-Order Structural Holes)
<p align="center">
  <img src="graphical abstract.jpg" width="800" alt="HOSH 算法框架图">
</p>
=========================================================

### 框架概览 (Framework Overview)

> **图 1：HOSH 算法逻辑示意图。** \* **(a) 核心单元**：将极大团作为网络分析的基础单元。 \* **(b) 评价体系**：通过相对自主性系数修正基础投资得到有效依赖，并量化由拓扑重叠产生的团间冗余。 \* **(c)-(e) 性能优势**：在空间分散度 (Dispersion)、排序分辨率 (Resolution) 和计算扩展性 (Scalability) 上表现卓越。
> 
> +4

* * *

1\. 系统要求 (System Requirements)
------------------------------

*   **Python 3.8+**
*   **必需的第三方库**：`networkx`, `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`, `openpyxl`

* * *

2\. 项目简介 (Overview)
-------------------

传统结构洞理论能有效识别复杂网络中的桥接节点；然而，现有方法主要依赖二元边交互视角，忽略了现实系统中普遍存在的高阶群体结构 。

**高阶结构洞 (HOSH)** 是一种基于极大团的新型关键节点识别方法 。它将拓扑分析的基础单元从单边升维至介观层面的**极大团 (Maximal Cliques)** 。通过整合资源依赖理论，HOSH 量化了节点对其所属团的**有效依赖**以及拓扑重叠产生的**团间冗余**，从而在高阶架构下重构了结构约束机制 。

+3

实验结果表明，HOSH 不仅能识别出空间分布高度离散的节点以最大化全局传播覆盖，还具备卓越的排序分辨率，平均单调性达 0.9901，并在稀疏网络中保持线性时间复杂度 。

+2

* * *

3\. 目录结构 (Repository Structure)
-------------------------------

```
HOSH_Project/
│
├── core/                           # 核心算法与数据处理模块
│   ├── hosh_methods.py             # HOSH算法及基准对比方法实现
│   ├── network_loader.py           # 自动化数据下载与LCC预处理工具
│   └── precompute_rankings.py      # 耗时排名评分的预计算与缓存引擎
│
├── experiments/                    # 多维度性能验证实验脚本
│   ├── exp_sir_influence.py        # 实验1: 稳态 SIR 传播规模分析
│   ├── exp_temporal_sir.py         # 实验2: 时序 SIR 传播演化曲线
│   ├── exp_improvement_rate.py     # 实验3: 算法改进率敏感性气泡图分析
│   ├── exp_monotonicity.py         # 实验4: 排序单调性 M(R) 统计分析
│   ├── exp_ranking_frequency.py    # 实验5: 排名数值简并频率分布可视化
│   ├── exp_spreader_separation.py  # 实验6: 传播者空间分散度雷达图分析
│   ├── exp_topology_visualization.py # 实验7: Top-50 节点网络拓扑分布可视化
│   ├── exp_parameter_sensitivity.py  # 实验8: HOSH 参数 ξ 鲁棒性 Kendall's Tau 验证
│   ├── exp_running_time.py         # 实验9: 真实网络运行时间对比
│   └── exp_synthetic_networks.py   # 实验10: 合成网络时间复杂度线性扩展验证
│
├── tools/                          # 辅助统计与验证工具
│   ├── compute_network_statistics.py # 计算网络基础拓扑特性 (生成 LaTeX 表)
│   └── verify_networks.py          # 数据集完整性与最大连通分量验证
│
├── networks_data/                  # 网络原始数据集存放目录 (由代码自动生成)
├── results/                        # 实验结果输出目录
│   ├── node_rankings/              # 预计算的评分缓存文件 (.pkl)
│   └── exp_*/                      # 各实验模块对应的输出子目录
│
├── 未标题-3.jpg                    # 算法框架概览图
└── README.md                       # 项目说明文档
```

* * *

4\. 文件详细说明 (File Descriptions)
------------------------------

### 4.1 核心算法与数据加载 (`core/`)

**Scripts:**

| 文件 (File) | 描述 (Description) |
| --- | --- |
| `hosh_methods.py` | **核心算法库**。实现了 HOSH 算法及 8 种基准算法（DC, BC, CC, K-Shell, CI, SH, ISH, SNC）的计算接口。 |
| `network_loader.py` | **数据预处理**。自动执行 SuiteSparse 库下载、无向化、去自环及提取最大连通分量 (LCC)。 |
| `precompute_rankings.py` | **缓存管理**。提前计算所有网络下各算法的节点分数并序列化保存，确保实验脚本高速读取。 |

**Data Files:**

| 文件 (File) | 描述 (Description) |
| --- | --- |
| `[network]_rankings.pkl` | 位于 `results/node_rankings/`，存储节点评分的字典对象，供绘图脚本快速调用。 |

### 4.2 实验评估与可视化 (`experiments/`)

**Scripts:**

| 文件 (File) | 描述 (Description) |
| --- | --- |
| `exp_sir_influence.py` | 评估不同种子比例  $p$  (2.5%-25%) 下的稳态感染规模  $F\left(t_{c}\right)$ 。 |
| `exp_temporal_sir.py` | 追踪选取 Top-10 节点作为初始种子时的时序传播演化轨迹。 |
| `exp_improvement_rate.py` | 在不同感染概率  $\beta /\beta _{th}$  下分析 HOSH 相对基准算法的改进率（气泡热力图）。 |
| `exp_monotonicity.py` | 计算排序单调性  $M\left(R\right)$ ，验证算法在细分节点差异上的表现。 |
| `exp_ranking_frequency.py` | 统计每个排名位置对应的节点数量，揭示排名并列问题。 |
| `exp_spreader_separation.py` | 计算种子节点的平均最短路径长度  $L_{s}$ ，反映传播者的空间覆盖能力。 |
| `exp_topology_visualization.py` | 将网络中 Top-50 的节点进行高亮渲染，展示独立子图对比。 |
| `exp_synthetic_networks.py` | 针对 BA, WS, ER 等合成网络（1k-20k规模）验证计算时间随节点数的增长趋势。 |

* * *

5\. 执行流水线 (Execution Pipeline)
------------------------------

请在项目根目录下按以下顺序运行脚本以复现论文实验：

```
Step 1: python -m tools.verify_networks          → 下载网络数据并验证 LCC 提取
          ↓
Step 2: python -m core.precompute_rankings       → 预计算节点排名 (极重要：生成评分缓存文件)
          ↓
Step 3: python -m experiments.exp_sir_influence  → 运行核心 SIR 影响力评估
          ↓
Step 4: python -m experiments.exp_monotonicity   # 计算分辨力指标并导出 Excel
          ↓
Step 5: python -m experiments.[其他实验脚本]      # 按需生成雷达图、拓扑可视化或复杂度曲线
```

> **运行说明**：本项目采用 Python 模块化导入模式。请确保在项目根目录下执行命令，并使用 `-m` 参数（如 `python -m core.hosh_methods`），以防出现相对路径导入错误。

* * *

6\. 注意事项 (Notes)
----------------

*   **评分缓存机制**：`experiments/` 下的脚本会优先加载 `results/node_rankings/` 中的 `.pkl` 文件。若修改了算法逻辑，需带 `--force` 参数重新运行 `precompute_rankings.py`。
*   **本地数据集**：`network_loader.py` 中标记为 `local` 的数据集（如 `infect`, `chesapeake`）需手动放置在 `networks_data/` 目录下。
*   **输出标准**：所有生成的图表均默认为 600 DPI 分辨率，采用 Times New Roman 字体，符合学术出版要求。

* * *

7\. 论文引用 (Citation)
-------------------

如果您在研究中使用了本代码或 HOSH 算法，请引用以下论文：

```
@article{YourLastName2025HOSH,
  title={Higher-Order Structural Holes: A Critical Node Identification Method Based on Maximal Cliques},
  author={Your Name and Co-authors},
  journal={TBD},
  year={2026}
}
```



---
Powered by [Gemini Exporter](https://www.ai-chat-exporter.com)