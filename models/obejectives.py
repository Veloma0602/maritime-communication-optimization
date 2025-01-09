import numpy as np
from .noise_model import NoiseModel

class ObjectiveFunction:
    def __init__(self, task_data, env_data, constraint_data):
        self.task_data = task_data
        self.env_data = env_data
        self.constraint_data = constraint_data
        self.noise_model = NoiseModel(env_data)
    
    def reliability_objective(self, x):
        """通信可靠性目标"""
        total_reliability = 0
        for i, params in enumerate(x):
            snr = self.calculate_snr(params, i)
            ber = self.calculate_bit_error_rate(snr, params['modulation'])
            packet_size = self.task_data['communication_links'][i]['packet_size']
            reliability = (1 - ber)**packet_size
            total_reliability += reliability
        return -total_reliability
    
    def spectral_efficiency_objective(self, x):
        """频谱效率目标"""
        total_efficiency = 0
        for i, params in enumerate(x):
            snr = self.calculate_snr(params, i)
            capacity = params['bandwidth'] * np.log2(1 + snr)
            efficiency = params['data_rate'] / capacity
            total_efficiency += efficiency
        return -total_efficiency
    
    def energy_efficiency_objective(self, x):
        """能量效率目标"""
        total_energy = 0
        for i, params in enumerate(x):
            successful_bits = (
                params['data_rate'] * 
                self.task_data['communication_links'][i]['duration'] *
                (1 - self.calculate_packet_loss_rate(params, i))
            )
            total_power = params['power'] + self.calculate_circuit_power(params)
            energy_per_bit = total_power / successful_bits
            total_energy += energy_per_bit
        return total_energy
    
    def calculate_snr(self, params, link_index):
        """计算信噪比"""
        link_params = self.task_data['communication_links'][link_index]
        noise = self.noise_model.calculate_total_noise(
            params['frequency'],
            link_params
        )
        return params['power'] - noise
    
    def calculate_bit_error_rate(self, snr, modulation):
        """计算误码率"""
        # 根据调制方式计算误码率
        if modulation == 'BPSK':
            return 0.5 * np.erfc(np.sqrt(snr))
        elif modulation == 'QPSK':
            return np.erfc(np.sqrt(snr/2))
        # 添加其他调制方式...
        return 0