from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
import numpy as np

class CommunicationProblem(Problem):
    def __init__(self, optimizer):
        super().__init__(
            n_var=optimizer.n_vars,
            n_obj=5,
            n_constr=optimizer.n_constraints,
            xl=optimizer.lower_bounds,
            xu=optimizer.upper_bounds
        )
        self.optimizer = optimizer
    
    def _evaluate(self, x, out, *args, **kwargs):
        f1 = [self.optimizer.objectives.reliability_objective(xi) for xi in x]
        f2 = [self.optimizer.objectives.spectral_efficiency_objective(xi) for xi in x]
        f3 = [self.optimizer.objectives.energy_efficiency_objective(xi) for xi in x]
        f4 = [self.optimizer.objectives.interference_objective(xi) for xi in x]
        f5 = [self.optimizer.objectives.adaptability_objective(xi) for xi in x]
        
        out["F"] = np.column_stack([f1, f2, f3, f4, f5])
        
        # 计算约束
        g = [self.optimizer.constraints.evaluate_constraints(xi) for xi in x]
        out["G"] = np.array(g)

class CommunicationOptimizer:
    def __init__(self, task_data, env_data, constraint_data, config):
        self.task_data = task_data
        self.env_data = env_data
        self.config = config
        self.objectives = ObjectiveFunction(task_data, env_data, constraint_data)
        self.constraints = Constraints(config)
        
        # 初始化问题维度
        self.setup_problem_dimensions()
    
    def setup_problem_dimensions(self):
        """设置问题维度和边界"""
        n_links = len(self.task_data['communication_links'])
        self.n_vars = n_links * 5  # 每个链路5个参数
        self.n_constraints = n_links * 7  # 每个链路7个约束
        
        # 设置边界
        self.lower_bounds = np.array([
            *[self.config.freq_min] * n_links,
            *[self.config.bandwidth_min] * n_links,
            *[self.config.power_min] * n_links,
            *[0] * n_links * 2  # 调制方式和极化方式的下界
        ])
        self.upper_bounds = np.array([
            *[self.config.freq_max] * n_links,
            *[self.config.bandwidth_max] * n_links,
            *[self.config.power_max] * n_links,
            *[3] * n_links * 2  # 调制方式和极化方式的上界
        ])
    
    def optimize(self):
        """运行优化过程"""
        problem = CommunicationProblem(self)
        
        algorithm = NSGA2(
            pop_size=self.config.population_size,
            n_offsprings=self.config.population_size,
            sampling=self.initialize_population,
            crossover=SBX(prob=self.config.crossover_prob, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        res = minimize(
            problem,
            algorithm,
            ('n_gen', self.config.n_generations),
            verbose=True
        )
        
        return res.F, res.X
    
    def initialize_population(self):
        """初始化种群，包括历史案例"""
        # 从Neo4j获取相似历史案例
        historical_cases = self.neo4j_handler.get_similar_cases(
            self.task_data['task_id'],
            limit=self.config.population_size // 2
        )
        
        # 创建初始种群
        population = []
        
        # 添加历史案例转换的解
        for case in historical_cases:
            solution = self.convert_case_to_solution(case)
            if self.is_valid_solution(solution):
                population.append(solution)
        
        # 补充随机生成的解
        while len(population) < self.config.population_size:
            solution = self.generate_random_solution()
            if self.is_valid_solution(solution):
                population.append(solution)
        
        return np.array(population)
    
    def convert_case_to_solution(self, case):
        """将历史案例转换为解向量"""
        solution = []
        for link in case['communication_links']:
            solution.extend([
                link['frequency'],
                link['bandwidth'],
                link['power'],
                self.modulation_to_index(link['modulation']),
                self.polarization_to_index(link['polarization'])
            ])
        return np.array(solution)
    
    def generate_random_solution(self):
        """生成随机解"""
        n_links = len(self.task_data['communication_links'])
        return np.random.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            size=self.n_vars
        )
    
    def is_valid_solution(self, solution):
        """检查解是否满足基本约束"""
        try:
            reshaped = solution.reshape(-1, 5)
            for link in reshaped:
                if not (self.config.freq_min <= link[0] <= self.config.freq_max):
                    return False
                if not (self.config.bandwidth_min <= link[1] <= self.config.bandwidth_max):
                    return False
                if not (self.config.power_min <= link[2] <= self.config.power_max):
                    return False
            return True
        except:
            return False