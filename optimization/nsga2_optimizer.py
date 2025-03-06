from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from .population import PopulationManager

class HistoricalCaseSampling(Sampling):
    """
    基于历史案例的采样策略
    """
    def __init__(self, population_manager, task_id, n_var, xl, xu, pop_size):
        super().__init__()
        self.population_manager = population_manager
        self.task_id = task_id
        self.n_var = n_var
        self.xl = xl
        self.xu = xu
        self.pop_size = pop_size

    def _do(self, problem, n_samples, **kwargs):
        return self.population_manager.initialize_population(
            self.task_id, 
            self.n_var, 
            self.xl, 
            self.xu, 
            n_samples
        )

class CommunicationProblem(Problem):
    def __init__(self, optimizer):
        """
        初始化通信优化问题
        
        参数:
        optimizer: 优化器实例
        """
        super().__init__(
            n_var=optimizer.n_vars,
            n_obj=5,
            n_constr=optimizer.n_constraints,
            xl=optimizer.lower_bounds,
            xu=optimizer.upper_bounds
        )
        self.optimizer = optimizer
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        评估解的目标函数值和约束
        
        参数:
        x: 解集合
        out: 输出字典
        """
        # 初始化目标函数值矩阵
        f1 = np.zeros(len(x))
        f2 = np.zeros(len(x))
        f3 = np.zeros(len(x))
        f4 = np.zeros(len(x))
        f5 = np.zeros(len(x))
        
        # 初始化约束矩阵
        g = np.zeros((len(x), self.n_constr))
        
        # 对每个解进行评估
        for i, xi in enumerate(x):
            # 将解向量转换为参数字典列表
            params_list = self.optimizer.population_manager.solution_to_parameters(
                xi, 
                self.optimizer.n_links
            )
            
            # 计算目标函数值
            f1[i] = self.optimizer.objectives.reliability_objective(params_list)
            f2[i] = self.optimizer.objectives.spectral_efficiency_objective(params_list)
            f3[i] = self.optimizer.objectives.energy_efficiency_objective(params_list)
            f4[i] = self.optimizer.objectives.interference_objective(params_list)
            f5[i] = self.optimizer.objectives.adaptability_objective(params_list)
            
            # 计算约束
            g[i] = self.optimizer.constraints.evaluate_constraints(params_list)
        
        # 将目标函数值合并为矩阵
        out["F"] = np.column_stack([f1, f2, f3, f4, f5])
        
        # 设置约束
        out["G"] = g

class CommunicationOptimizer:
    def __init__(self, task_data: Dict, env_data: Dict, constraint_data: Dict, 
                config: Any, neo4j_handler=None):
        """
        初始化通信优化器
        
        参数:
        task_data: 任务数据
        env_data: 环境数据
        constraint_data: 约束数据
        config: 优化配置
        neo4j_handler: Neo4j数据库处理器
        """
        self.task_data = task_data
        self.env_data = env_data
        self.constraint_data = constraint_data
        self.config = config
        self.neo4j_handler = neo4j_handler
        
        # 初始化目标函数和约束
        from models.objectives import ObjectiveFunction
        from models.constraints import Constraints
        
        self.objectives = ObjectiveFunction(task_data, env_data, constraint_data)
        self.constraints = Constraints(config)
        
        # 初始化种群管理器
        self.population_manager = PopulationManager(config, neo4j_handler)
        
        # 设置问题维度
        self.setup_problem_dimensions()
    
    def setup_problem_dimensions(self):
        """设置问题维度和边界"""
        # 获取通信链路数量
        self.n_links = len(self.task_data['communication_links'])
        print(f"当前任务包含 {self.n_links} 个通信链路")
        
        # 设置问题维度：每个链路有5个参数
        self.n_vars = self.n_links * 5
        
        # 设置约束数量：每个链路有多个约束
        self.n_constraints = self.n_links * 7
        
        # 设置边界
        self.lower_bounds = np.array([
            *[self.config.freq_min] * self.n_links,            # 频率下界
            *[self.config.bandwidth_min] * self.n_links,       # 带宽下界
            *[self.config.power_min] * self.n_links,           # 功率下界
            *[0] * self.n_links,                               # 调制方式下界
            *[0] * self.n_links                                # 极化方式下界
        ])
        
        self.upper_bounds = np.array([
            *[self.config.freq_max] * self.n_links,            # 频率上界
            *[self.config.bandwidth_max] * self.n_links,       # 带宽上界
            *[self.config.power_max] * self.n_links,           # 功率上界
            *[3] * self.n_links,                               # 调制方式上界
            *[3] * self.n_links                                # 极化方式上界
        ])
        
        print(f"优化问题维度: {self.n_vars}, 约束数量: {self.n_constraints}")
    
    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        运行优化过程
        
        返回:
        Tuple[np.ndarray, np.ndarray]: Pareto前沿和对应的解
        """
        # 创建问题实例
        problem = CommunicationProblem(self)
        
        # 使用历史案例初始化种群
        task_id = self.task_data.get('task_info', {}).get('task_id', 'unknown')
        print(f"为任务 {task_id} 启动优化过程")
        
        sampling = HistoricalCaseSampling(
            self.population_manager,
            task_id,
            self.n_vars,
            self.lower_bounds,
            self.upper_bounds,
            self.config.population_size
        )
        
        # 配置NSGA-II算法
        algorithm = NSGA2(
            pop_size=self.config.population_size,
            n_offsprings=self.config.population_size,
            sampling=sampling,
            crossover=SBX(prob=self.config.crossover_prob, eta=15),
            mutation=PM(prob=self.config.mutation_prob, eta=20),
            eliminate_duplicates=True
        )
        
        # 运行优化
        print(f"开始运行NSGA-II算法，共 {self.config.n_generations} 代")
        res = minimize(
            problem,
            algorithm,
            ('n_gen', self.config.n_generations),
            verbose=True
        )
        
        print(f"优化完成，找到 {len(res.F)} 个非支配解")
        
        # 返回Pareto前沿和对应的解
        return res.F, res.X
    
    def post_process_solutions(self, objectives: np.ndarray, variables: np.ndarray) -> List[Dict]:
        """
        对优化结果进行后处理
        
        参数:
        objectives: 目标函数值
        variables: 对应的解变量
        
        返回:
        处理后的结果列表
        """
        results = []
        
        for i, (obj, var) in enumerate(zip(objectives, variables)):
            # 将解向量转换为参数
            params = self.population_manager.solution_to_parameters(var, self.n_links)
            
            # 创建结果字典
            result = {
                'solution_id': i,
                'objectives': {
                    'reliability': float(-obj[0]),  # 注意取反，因为优化过程中是最小化目标
                    'spectral_efficiency': float(-obj[1]),
                    'energy_efficiency': float(-obj[2]),
                    'interference': float(-obj[3]),
                    'adaptability': float(-obj[4])
                },
                'parameters': params
            }
            
            # 计算加权目标
            weighted_obj = (
                self.config.reliability_weight * (-obj[0]) +
                self.config.spectral_weight * (-obj[1]) +
                self.config.energy_weight * (-obj[2]) +
                self.config.interference_weight * (-obj[3]) +
                self.config.adaptability_weight * (-obj[4])
            )
            result['weighted_objective'] = float(weighted_obj)
            
            results.append(result)
        
        # 按加权目标排序
        results.sort(key=lambda x: x['weighted_objective'], reverse=True)
        
        return results