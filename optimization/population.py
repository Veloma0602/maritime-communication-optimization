import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import random

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
        try:
            # 每个链路需要5个参数，重塑解向量
            solution_reshaped = solution.reshape(-1)  # 先展平为1维
            
            # 确保解向量长度至少为n_links*5
            if len(solution_reshaped) < n_links * 5:
                # 如果解向量不够长，使用默认值填充
                padding = np.array([4e9, 20e6, 10, 0, 0] * (n_links - len(solution_reshaped) // 5))
                solution_reshaped = np.concatenate([solution_reshaped, padding])
                
            # 重新塑造为正确的维度
            solution_reshaped = solution_reshaped[:n_links*5].reshape(n_links, 5)
            
            for i in range(n_links):
                if i < len(solution_reshaped):
                    # 应用参数约束确保合理值
                    freq = self._constrain_param(solution_reshaped[i, 0], self.config.freq_min, self.config.freq_max)
                    bandwidth = self._constrain_param(solution_reshaped[i, 1], self.config.bandwidth_min, self.config.bandwidth_max)
                    power = self._constrain_param(solution_reshaped[i, 2], self.config.power_min, self.config.power_max)
                    
                    params = {
                        'frequency': freq,
                        'bandwidth': bandwidth,
                        'power': power,
                        'modulation': self.index_to_modulation(solution_reshaped[i, 3]),
                        'polarization': self.index_to_polarization(solution_reshaped[i, 4])
                    }
                    params_list.append(params)
        except Exception as e:
            print(f"解向量转换失败: {str(e)}")
            # 如果转换失败，创建默认参数
            for i in range(n_links):
                params = {
                    'frequency': 4e9,  # 4 GHz
                    'bandwidth': 20e6,  # 20 MHz
                    'power': 10,       # 10 W
                    'modulation': 'BPSK',
                    'polarization': 'LINEAR'
                }
                params_list.append(params)
        
        return params_list
    
    def _constrain_param(self, value: float, min_val: float, max_val: float) -> float:
        """
        将参数值约束在指定范围内
        
        参数:
        value: 参数值
        min_val: 最小允许值
        max_val: 最大允许值
        
        返回:
        约束后的参数值
        """
        if value < min_val or value > max_val:
            # 如果大幅超出范围，使用合理的默认值
            if value < min_val * 0.5 or value > max_val * 2:
                return (min_val + max_val) / 2
            # 否则剪裁到边界
            return max(min_val, min(value, max_val))
        return value

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
        生成随机解 - 改进版
        
        参数:
        n_solutions: 需要生成的解的数量
        problem_size: 问题维度
        lower_bounds: 下界
        upper_bounds: 上界
        
        返回:
        随机生成的解列表
        """
        solutions = []
        n_links = problem_size // 5  # 每个链路5个参数
        
        # 常用频段和带宽配置
        freq_bands = [
            (400e6, 500e6),    # UHF频段
            (1.5e9, 1.6e9),    # L频段
            (2.3e9, 2.5e9),    # S频段
            (3.4e9, 3.8e9),    # C频段
            (4.8e9, 5.2e9)     # C频段
        ]
        
        bandwidth_options = [5e6, 10e6, 20e6, 40e6, 50e6]  # 常用带宽选项
        
        for _ in range(n_solutions):
            # 为每个链路生成合理参数
            solution = []
            
            # 确保不同链路使用不同频段
            available_bands = freq_bands.copy()
            random.shuffle(available_bands)
            
            for i in range(n_links):
                # 1. 频率 - 从可用频段中选择
                if i < len(available_bands):
                    band = available_bands[i]
                    freq = random.uniform(band[0], band[1])
                else:
                    # 如果链路数量大于预设频段数量，则随机选择频段
                    band = random.choice(freq_bands)
                    freq = random.uniform(band[0], band[1])
                solution.append(freq)
                
                # 2. 带宽 - 根据频率选择合适的带宽
                if freq < 1e9:
                    bw = random.choice(bandwidth_options[:2])  # 低频段使用较小带宽
                elif freq < 3e9:
                    bw = random.choice(bandwidth_options[1:3])  # 中频段使用中等带宽
                else:
                    bw = random.choice(bandwidth_options[2:])  # 高频段使用较大带宽
                solution.append(bw)
                
                # 3. 功率 - 根据频率选择合适的功率
                if freq < 1e9:
                    power = random.uniform(30, 60)  # 低频段使用较大功率
                elif freq < 3e9:
                    power = random.uniform(20, 40)  # 中频段使用中等功率
                else:
                    power = random.uniform(5, 25)   # 高频段使用较小功率
                solution.append(power)
                
                # 4. 调制方式 - 根据频率和带宽选择合适的调制方式
                if freq < 1e9 or bw < 10e6:
                    mod = random.randint(0, 1)  # 低频/小带宽使用简单调制(BPSK/QPSK)
                else:
                    mod = random.randint(0, 3)  # 高频/大带宽可使用复杂调制
                solution.append(mod)
                
                # 5. 极化方式 - 根据应用场景选择
                if freq < 1e9:
                    pol = 0  # 低频段常用线性极化
                elif freq > 3e9:
                    pol = 1  # 高频段常用圆极化
                else:
                    pol = random.randint(0, 3)  # 中频段可使用各种极化
                solution.append(pol)
            
            # 将解转换为numpy数组
            sol_array = np.array(solution)
            
            # 检查解是否在边界内，如果不在则截断
            sol_array = np.clip(sol_array, lower_bounds, upper_bounds)
            
            solutions.append(sol_array)
        
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
            
            # 检查频率间隔
            # 每个链路需要5个参数，前5*n_links个元素表示频率
            n_links = len(solution) // 5
            
            # 获取所有链路的频率和带宽
            frequencies = solution[:n_links]
            bandwidths = solution[n_links:2*n_links]
            
            # 检查频率间隔是否满足要求
            min_spacing = np.max(bandwidths) * 1.2  # 最小间隔为最大带宽的1.2倍
            
            for i in range(n_links):
                for j in range(i+1, n_links):
                    spacing = abs(frequencies[i] - frequencies[j])
                    edge_spacing = spacing - (bandwidths[i]/2 + bandwidths[j]/2)
                    
                    if edge_spacing < min_spacing:
                        return False
            
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