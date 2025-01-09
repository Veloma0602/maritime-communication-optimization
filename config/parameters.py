class OptimizationConfig:
    def __init__(self):
        # NSGA-II参数
        self.population_size = 100
        self.n_generations = 200
        self.mutation_prob = 0.1
        self.crossover_prob = 0.9
        
        # 频率范围约束
        self.freq_min = 100e6  # 100MHz
        self.freq_max = 10e9   # 10GHz
        
        # 功率约束
        self.power_min = 0.1   # W
        self.power_max = 100   # W
        
        # 带宽约束
        self.bandwidth_min = 1e6    # 1MHz
        self.bandwidth_max = 100e6  # 100MHz
        
        # 信噪比约束
        self.snr_min = 10  # dB
        
        # 时延约束
        self.delay_max = 100  # ms
        
        # 权重参数
        self.reliability_weight = 0.3
        self.spectral_weight = 0.2
        self.energy_weight = 0.2
        self.interference_weight = 0.15
        self.adaptability_weight = 0.15