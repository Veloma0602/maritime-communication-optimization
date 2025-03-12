class OptimizationConfig:
    def __init__(self):
        """初始化优化配置参数"""
        # NSGA-II算法参数
        self.population_size = 100       # 种群大小
        self.n_generations = 200         # 迭代代数
        self.mutation_prob = 0.1         # 变异概率
        self.crossover_prob = 0.9        # 交叉概率
        
        # 频率范围约束 (Hz)
        self.freq_min = 100e6            # 100 MHz
        self.freq_max = 10e9             # 10 GHz
        
        # 功率约束 (W)
        self.power_min = 5.0             # 最小功率
        self.power_max = 50            # 最大功率
        
        # 带宽约束 (Hz)
        self.bandwidth_min = 5e6         # 5 MHz
        self.bandwidth_max = 50e6       # 50 MHz
        
        # 信噪比约束 (dB)
        self.snr_min = 10                # 最小信噪比
        
        # 时延约束 (ms)
        self.delay_max = 100             # 最大时延
        
        # 频率间隔约束 (Hz)
        self.min_freq_separation = 5e6   # 最小频率间隔
        
        # 目标权重
        self.reliability_weight = 0.3    # 可靠性权重
        self.spectral_weight = 0.2       # 频谱效率权重
        self.energy_weight = 0.2         # 能量效率权重
        self.interference_weight = 0.15  # 抗干扰权重
        self.adaptability_weight = 0.15  # 环境适应性权重
        
        # 历史案例使用配置
        self.use_historical_cases = True # 是否使用历史案例
        self.max_historical_cases = 20   # 最大历史案例数量
        self.historical_ratio = 0.4      # 历史案例在初始种群中的比例
        
        # 多目标搜索配置
        self.use_reference_points = True # 是否使用参考点引导搜索
        self.ref_point_update_freq = 20  # 参考点更新频率（代数）
        
        # 演化操作符参数
        self.crossover_eta = 15          # 交叉分布指数
        self.mutation_eta = 20           # 变异分布指数
        
        # 结果保存配置
        self.save_intermediate = True    # 是否保存中间结果
        self.save_frequency = 50         # 保存频率（代数）
    
    def set_population_size(self, size: int):
        """设置种群大小"""
        if size > 0:
            self.population_size = size
            return True
        return False
    
    def set_generations(self, generations: int):
        """设置迭代代数"""
        if generations > 0:
            self.n_generations = generations
            return True
        return False
    
    def set_mutation_probability(self, prob: float):
        """设置变异概率"""
        if 0 <= prob <= 1:
            self.mutation_prob = prob
            return True
        return False
    
    def set_crossover_probability(self, prob: float):
        """设置交叉概率"""
        if 0 <= prob <= 1:
            self.crossover_prob = prob
            return True
        return False
    
    def set_frequency_range(self, min_freq: float, max_freq: float):
        """设置频率范围"""
        if 0 < min_freq < max_freq:
            self.freq_min = min_freq
            self.freq_max = max_freq
            return True
        return False
    
    def set_power_range(self, min_power: float, max_power: float):
        """设置功率范围"""
        if 0 < min_power < max_power:
            self.power_min = min_power
            self.power_max = max_power
            return True
        return False
    
    def set_bandwidth_range(self, min_bw: float, max_bw: float):
        """设置带宽范围"""
        if 0 < min_bw < max_bw:
            self.bandwidth_min = min_bw
            self.bandwidth_max = max_bw
            return True
        return False
    
    def set_objective_weights(self, weights: dict):
        """设置目标权重"""
        if not weights:
            return False
            
        if 'reliability' in weights:
            self.reliability_weight = weights['reliability']
            
        if 'spectral' in weights:
            self.spectral_weight = weights['spectral']
            
        if 'energy' in weights:
            self.energy_weight = weights['energy']
            
        if 'interference' in weights:
            self.interference_weight = weights['interference']
            
        if 'adaptability' in weights:
            self.adaptability_weight = weights['adaptability']
            
        # 归一化权重
        total = (self.reliability_weight + self.spectral_weight + 
                self.energy_weight + self.interference_weight + 
                self.adaptability_weight)
                
        if total > 0:
            self.reliability_weight /= total
            self.spectral_weight /= total
            self.energy_weight /= total
            self.interference_weight /= total
            self.adaptability_weight /= total
            return True
            
        return False
    
    def to_dict(self):
        """将配置转换为字典"""
        return {
            "population_size": self.population_size,
            "n_generations": self.n_generations,
            "mutation_prob": self.mutation_prob,
            "crossover_prob": self.crossover_prob,
            "freq_min": self.freq_min,
            "freq_max": self.freq_max,
            "power_min": self.power_min,
            "power_max": self.power_max,
            "bandwidth_min": self.bandwidth_min,
            "bandwidth_max": self.bandwidth_max,
            "snr_min": self.snr_min,
            "delay_max": self.delay_max,
            "min_freq_separation": self.min_freq_separation,
            "objective_weights": {
                "reliability": self.reliability_weight,
                "spectral": self.spectral_weight,
                "energy": self.energy_weight,
                "interference": self.interference_weight,
                "adaptability": self.adaptability_weight
            },
            "use_historical_cases": self.use_historical_cases,
            "max_historical_cases": self.max_historical_cases,
            "historical_ratio": self.historical_ratio
        }