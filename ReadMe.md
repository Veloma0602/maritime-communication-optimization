# Maritime Communication Parameters Optimization

基于 NSGA-II 算法的海上通信参数多目标优化系统，用于优化舰船通信方案中的各项通信参数，实现通信可靠性、频谱效率、能量效率等多个目标的综合优化。

## 功能特点

- 基于 NSGA-II 的多目标优化
- 完整的海上通信噪声模型
- 支持历史案例的复用
- 与 Neo4j 图数据库集成
- 考虑实际海上环境因素
- 支持多种通信约束条件

## 系统要求

- Python 3.8+
- Neo4j 4.4+
- 足够的计算资源用于优化过程

## 安装指南

1. 克隆仓库：
```bash
git clone [repository-url]
cd maritime-communication-optimization
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 配置说明

1. Neo4j 数据库配置：
   - 在 `config/parameters.py` 中设置数据库连接参数
   - 确保 Neo4j 服务器正在运行

2. 优化参数配置：
   - 在 `config/parameters.py` 中调整优化算法参数
   - 可根据实际需求修改约束条件

## 使用方法

1. 准备数据：
   - 确保任务数据已导入 Neo4j
   - 检查环境条件和约束条件的完整性

2. 运行优化：
```bash
python main.py --task-id rw001
```

3. 查看结果：
   - 优化结果会自动保存到 Neo4j 数据库
   - 使用 Neo4j Browser 查看结果

## 项目结构

```
.
├── main.py                     # 主程序入口
├── config/                     # 配置文件
│   ├── __init__.py
│   └── parameters.py           # 参数配置
├── models/                     # 模型实现
│   ├── __init__.py
│   ├── objectives.py          # 目标函数
│   ├── constraints.py         # 约束条件
│   └── noise_model.py         # 噪声模型
├── optimization/              # 优化算法
│   ├── __init__.py
│   ├── nsga2_optimizer.py    # NSGA-II 优化器
│   └── population.py         # 种群管理
├── data/                     # 数据处理
│   ├── __init__.py
│   └── neo4j_handler.py      # Neo4j 接口
└── utils/                    # 工具函数
    ├── __init__.py
    └── validators.py         # 数据验证
```


## 核心模块说明

### 数据层 (data/)
**Neo4jHandler类**
- 作用：处理与Neo4j数据库的所有交互
- 主要方法：
  - get_task_data(): 获取任务相关数据
  - get_environment_data(): 获取环境条件数据
  - get_constraint_data(): 获取通信约束数据
  - get_similar_cases(): 获取相似历史案例
  - save_optimization_results(): 保存优化结果

### 模型层 (models/)
**NoiseModel类**
- 作用：实现完整的噪声计算模型
- 主要计算：
  - 环境噪声（海浪、雨噪声等）
  - 多径噪声
  - 传播损耗

**ObjectiveFunction类**
- 作用：实现五个优化目标的计算
- 目标函数：
  1. 通信可靠性
  2. 频谱效率
  3. 能量效率
  4. 抗干扰性能
  5. 环境适应性

**Constraints类**
- 作用：处理所有约束条件
- 主要约束：
  - 频率范围约束
  - 功率约束
  - 带宽约束
  - 信噪比约束
  - 时延约束

### 优化层 (optimization/)
**CommunicationOptimizer类**
- 作用：主要的优化器实现
- 核心功能：
  - 配置NSGA-II算法
  - 管理优化过程
  - 处理结果

**PopulationManager类**
- 作用：管理优化算法的种群
- 主要功能：
  - 从历史案例初始化种群
  - 生成随机解
  - 验证解的可行性

## 执行流程
1. 从Neo4j获取任务数据
2. 初始化优化器
3. 执行NSGA-II优化
4. 保存结果到Neo4j


## 优化目标

1. 通信可靠性优化
   - 最小化误码率
   - 保证信噪比要求

2. 频谱效率优化
   - 优化带宽利用率
   - 减少频谱浪费

3. 能量效率优化
   - 优化功率分配
   - 减少能量消耗

4. 抗干扰性能优化
   - 考虑多径效应
   - 降低互干扰

5. 环境适应性优化
   - 适应海况变化
   - 考虑电磁环境

## 示例查询

查看优化结果：
```cypher
MATCH (t:任务 {任务编号: 'rw001'})-[:优化方案]->(r:优化结果)
RETURN r.结果编号, r.可靠性目标, r.频谱效率目标
ORDER BY r.可靠性目标 DESC
LIMIT 5;
```

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交改动
4. 发起 Pull Request

## 许可证

MIT License

## 联系方式

- 项目维护者：[Your Name]
- 邮箱：[Your Email]

## 参考文献

1. [NSGA-II: A multi-objective optimization algorithm](reference-url)
2. [Maritime Communication Systems](reference-url)
3. [Neo4j Graph Database](reference-url)

