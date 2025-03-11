import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class PopulationManager:
    def __init__(self, config, neo4j_handler=None):
        """
        初始化种群管理器
        
        参数:
        config: 优化配置
        neo4j_handler: Neo4j数据库处理器，用于获取历史案例
        """
        self.config = config
        self.neo4j_handler = neo4j_handler        

    def initialize_population(self, task_id: str, problem_size: int,
                            lower_bounds: np.ndarray, upper_bounds: np.ndarray,
                            n_population: int) -> np.ndarray:
        """
        初始化种群，融合历史案例和随机生成的解
        
        参数:
        task_id: 当前任务ID
        problem_size: 问题维度
        lower_bounds: 下界
        upper_bounds: 上界
        n_population: 种群大小
        
        返回:
        初始化的种群
        """
        population = []
        
        # 获取相似历史案例
        historical_cases = []
        if self.neo4j_handler:
            try:
                historical_cases = self.neo4j_handler.get_similar_cases(
                    task_id, 
                    limit=min(n_population // 2, 10)  # 最多取一半种群大小的历史案例
                )
                print(f"找到 {len(historical_cases)} 个相似历史案例用于初始化种群")
            except Exception as e:
                print(f"获取相似历史案例时出错: {str(e)}")
                historical_cases = []
        
        # 从历史案例生成初始解
        for case_id in historical_cases:
            try:
                # 获取该任务的通信链路
                communication_links = self.neo4j_handler.get_task_communication_links(case_id)
                
                if communication_links:
                    # 提取参数并转换为解向量
                    parameters = []
                    for link in communication_links:
                        params = self.neo4j_handler._extract_communication_parameters(link)
                        if params:
                            parameters.append(params)
                    
                    # 转换为解向量
                    if parameters:
                        solution = self._parameters_to_solution(parameters)
                        
                        # 如果解向量维度不匹配，进行调整
                        if len(solution) > problem_size:
                            # 截断
                            solution = solution[:problem_size]
                        elif len(solution) < problem_size:
                            # 扩展（用随机值填充）
                            padding = np.random.uniform(
                                low=lower_bounds[len(solution):],
                                high=upper_bounds[len(solution):],
                                size=problem_size - len(solution)
                            )
                            solution = np.concatenate([solution, padding])
                        
                        # 验证解是否有效
                        if self.validate_solution(solution, lower_bounds, upper_bounds):
                            population.append(solution)
                            print(f"从任务 {case_id} 的通信参数生成有效解")
                        
                        # 如果已经收集了足够的解，就停止
                        if len(population) >= n_population // 2:
                            break
            except Exception as e:
                print(f"处理历史案例 {case_id} 时出错: {str(e)}")
                continue
        
        # 如果历史案例生成的解太少或没有历史案例，生成随机解补充种群
        n_random = n_population - len(population)
        if n_random > 0:
            print(f"生成 {n_random} 个随机解补充种群")
            try:
                random_solutions = self.generate_random_solutions(
                    n_random,
                    problem_size,
                    lower_bounds,
                    upper_bounds
                )
                population.extend(random_solutions)
            except Exception as e:
                print(f"生成随机解时出错: {str(e)}")
        
        # 确保种群大小正确
        if len(population) < n_population:
            print(f"种群大小不足 ({len(population)}/{n_population})，生成额外随机解")
            try:
                # 简单的随机生成，不依赖于 generate_random_solutions
                for _ in range(n_population - len(population)):
                    solution = np.random.uniform(
                        low=lower_bounds,
                        high=upper_bounds
                    )
                    population.append(solution)
            except Exception as e:
                print(f"生成额外随机解时出错: {str(e)}")
                # 如果所有方法都失败，使用一个默认值填充
                while len(population) < n_population:
                    default_solution = (lower_bounds + upper_bounds) / 2
                    population.append(default_solution)
        
        return np.array(population)
    
    def generate_random_solutions(self, n_solutions: int, problem_size: int,
                                 lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> List[np.ndarray]:
        """
        生成随机解
        
        参数:
        n_solutions: 需要生成的解的数量
        problem_size: 问题维度
        lower_bounds: 下界
        upper_bounds: 上界
        
        返回:
        随机生成的解列表
        """
        solutions = []
        attempts = 0
        max_attempts = n_solutions * 10  # 最大尝试次数
        
        while len(solutions) < n_solutions and attempts < max_attempts:
            solution = np.random.uniform(
                low=lower_bounds,
                high=upper_bounds,
                size=problem_size
            )
            
            if self.validate_solution(solution, lower_bounds, upper_bounds):
                solutions.append(solution)
            
            attempts += 1
            
        # 如果尝试次数过多仍未生成足够的解，放宽验证条件
        if len(solutions) < n_solutions:
            print(f"警告: 尝试 {attempts} 次后仅生成 {len(solutions)}/{n_solutions} 个有效解，将放宽验证条件")
            while len(solutions) < n_solutions:
                solution = np.random.uniform(
                    low=lower_bounds,
                    high=upper_bounds,
                    size=problem_size
                )
                solutions.append(solution)
        
        return solutions
    
    def validate_solution(self, solution: np.ndarray, 
                         lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> bool:
        """
        验证解的可行性
        
        参数:
        solution: 待验证的解
        lower_bounds: 下界
        upper_bounds: 上界
        
        返回:
        解是否有效
        """
        try:
            # 检查基本边界约束
            if np.any(solution < lower_bounds) or np.any(solution > upper_bounds):
                return False
            
            # 附加的有效性检查可以在这里添加
            # 例如检查频率间隔、功率分配等
            
            return True
        except Exception as e:
            print(f"解验证出错: {str(e)}")
            return False

    def convert_case_to_solution(self, case_data: Dict[str, Any], problem_size: int) -> Optional[np.ndarray]:
        """
        将历史案例转换为解向量
        
        参数:
        case_data: 历史案例数据
        problem_size: 问题维度
        
        返回:
        解向量或None（如果转换失败）
        """
        try:
            # 案例中应该包含通信链路的参数
            solution = []
            comm_links = case_data.get('communication_links', [])
            
            for link in comm_links:
                # 提取每个链路的参数
                freq = float(link.get('frequency', 0))
                bandwidth = float(link.get('bandwidth', 0))
                power = float(link.get('power', 0))
                modulation = self.modulation_to_index(link.get('modulation', ''))
                polarization = self.polarization_to_index(link.get('polarization', ''))
                
                solution.extend([freq, bandwidth, power, modulation, polarization])
            
            # 检查维度是否匹配
            if len(solution) != problem_size:
                print(f"警告: 解向量维度不匹配 (实际 {len(solution)}, 预期 {problem_size})")
                return None
            
            return np.array(solution)
            
        except Exception as e:
            print(f"转换案例到解向量时出错: {str(e)}")
            return None
    
    def modulation_to_index(self, modulation: str) -> float:
        """将调制方式转换为数值索引"""
        modulation_map = {
            'BPSK': 0.0,
            'QPSK': 1.0,
            'QAM16': 2.0,
            'QAM64': 3.0
        }
        return modulation_map.get(modulation.upper(), 0.0)
    
    def polarization_to_index(self, polarization: str) -> float:
        """将极化方式转换为数值索引"""
        polarization_map = {
            'LINEAR': 0.0,
            'CIRCULAR': 1.0,
            'DUAL': 2.0,
            'ADAPTIVE': 3.0
        }
        return polarization_map.get(polarization.upper(), 0.0)
    
    def index_to_modulation(self, index: float) -> str:
        """将数值索引转换为调制方式"""
        index_map = {
            0: 'BPSK',
            1: 'QPSK',
            2: 'QAM16',
            3: 'QAM64'
        }
        return index_map.get(int(round(index)), 'BPSK')
    
    def index_to_polarization(self, index: float) -> str:
        """将数值索引转换为极化方式"""
        index_map = {
            0: 'LINEAR',
            1: 'CIRCULAR',
            2: 'DUAL',
            3: 'ADAPTIVE'
        }
        return index_map.get(int(round(index)), 'LINEAR')
    
    def solution_to_parameters(self, solution: np.ndarray, n_links: int) -> List[Dict[str, Any]]:
        """
        将解向量转换为通信参数字典列表
        
        参数:
        solution: 解向量
        n_links: 通信链路数量
        
        返回:
        参数字典列表
        """
        params_list = []
        solution = solution.reshape(-1, 5)  # 每个链路5个参数
        
        for i in range(n_links):
            if i < len(solution):
                params = {
                    'frequency': solution[i, 0],
                    'bandwidth': solution[i, 1],
                    'power': solution[i, 2],
                    'modulation': self.index_to_modulation(solution[i, 3]),
                    'polarization': self.index_to_polarization(solution[i, 4])
                }
                params_list.append(params)
        
        return params_list
    
    def _parameters_to_solution(self, parameters: List[Dict]) -> np.ndarray:
        """

        将通信参数列表转换为解向量

        """
        solution = []
        
        for params in parameters:
            # 添加频率
            solution.append(params.get('frequency', 0))
            
            # 添加带宽
            solution.append(params.get('bandwidth', 0))
            
            # 添加功率
            solution.append(params.get('power', 0))
            
            # 添加调制方式(转换为数值索引)
            modulation = params.get('modulation', 'BPSK')
            solution.append(self.modulation_to_index(modulation))
            
            # 添加极化方式(转换为数值索引)
            polarization = params.get('polarization', 'LINEAR')
            solution.append(self.polarization_to_index(polarization))
        
        return np.array(solution)