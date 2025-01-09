import numpy as np

class Constraints:
    def __init__(self, config):
        self.config = config
    
    def evaluate_constraints(self, x):
        """评估所有约束条件"""
        constraints = []
        
        # 频率约束
        freq_constraints = self.frequency_constraints(x)
        constraints.extend(freq_constraints)
        
        # 功率约束
        power_constraints = self.power_constraints(x)
        constraints.extend(power_constraints)
        
        # 带宽约束
        bandwidth_constraints = self.bandwidth_constraints(x)
        constraints.extend(bandwidth_constraints)
        
        # 信噪比约束
        snr_constraints = self.snr_constraints(x)
        constraints.extend(snr_constraints)
        
        # 时延约束
        delay_constraints = self.delay_constraints(x)
        constraints.extend(delay_constraints)
        
        return np.array(constraints)
    
    def frequency_constraints(self, x):
        """频率约束"""
        constraints = []
        for params in x:
            c1 = self.config.freq_min - params['frequency']
            c2 = params['frequency'] - self.config.freq_max
            constraints.extend([c1, c2])
        return constraints
    
    def power_constraints(self, x):
        """功率约束"""
        constraints = []
        for params in x:
            c = params['power'] - self.config.power_max
            constraints.append(c)
        return constraints