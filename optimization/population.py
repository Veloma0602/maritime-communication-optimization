import numpy as np
from typing import List, Dict

class PopulationManager:
    def __init__(self, config):
        self.config = config
        
    def initialize_population(self, historical_cases: List[Dict],
                            problem_size: int) -> np.ndarray:
        """初始化种群"""
        population = []
        
        # 从历史案例生成初始解
        historical_solutions = self.process_historical_cases(
            historical_cases,
            problem_size
        )
        population.extend(historical_solutions)
        
        # 生成随机解补充种群
        n_random = self.config.population_size - len(population)
        random_solutions = self.generate_random_solutions(
            n_random,
            problem_size
        )
        population.extend(random_solutions)
        
        return np.array(population)
    
    def process_historical_cases(self, cases: List[Dict],
                               problem_size: int) -> List[np.ndarray]:
        """处理历史案例"""
        solutions = []
        for case in cases:
            try:
                solution = self.convert_case_to_solution(case, problem_size)
                if self.validate_solution(solution):
                    solutions.append(solution)
            except:
                continue
        return solutions
    
    def generate_random_solutions(self, n_solutions: int,
                                problem_size: int) -> List[np.ndarray]:
        """生成随机解"""
        solutions = []
        while len(solutions) < n_solutions:
            solution = np.random.uniform(
                low=self.config.lower_bounds,
                high=self.config.upper_bounds,
                size=problem_size
            )
            if self.validate_solution(solution):
                solutions.append(solution)
        return solutions
    
    def validate_solution(self, solution: np.ndarray) -> bool:
        """验证解的可行性"""
        try:
            # 检查基本约束
            if not all(self.config.freq_min <= solution[::5] <= self.config.freq_max):
                return False
            if not all(self.config.power_min <= solution[2::5] <= self.config.power_max):
                return False
            
            # 检查频率间隔
            frequencies = solution[::5]
            for i in range(len(frequencies)):
                for j in range(i + 1, len(frequencies)):
                    if abs(frequencies[i] - frequencies[j]) < self.config.min_freq_separation:
                        return False
            
            return True
        except:
            return False