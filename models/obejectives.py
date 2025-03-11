import numpy as np
from typing import List, Dict, Any
from .noise_model import NoiseModel

class ObjectiveFunction:
    def __init__(self, task_data: Dict, env_data: Dict, constraint_data: Dict):
        """
        初始化目标函数
        
        参数:
        task_data: 任务数据
        env_data: 环境数据
        constraint_data: 约束数据
        """
        self.task_data = task_data
        self.env_data = env_data
        self.constraint_data = constraint_data
        self.noise_model = NoiseModel(env_data)
        
        # 初始化额外的参数
        self._setup_params()

    def _setup_params(self):
        """设置计算所需的参数"""
        # 从环境数据中提取信息
        self.sea_state = self._parse_numeric_value(self.env_data.get('海况等级', 3))
        self.emi_intensity = self._parse_numeric_value(self.env_data.get('电磁干扰强度', 0.5))
        
        # 解析背景噪声 (如 "-107dBm")
        self.background_noise = self._parse_numeric_value(self.env_data.get('背景噪声', -100))
        
        # 从约束数据中提取信息
        self.min_reliability = self._parse_numeric_value(self.constraint_data.get('最小可靠性要求', 0.95))
        self.max_delay = self._parse_numeric_value(self.constraint_data.get('最大时延要求', 100))
        self.min_snr = self._parse_numeric_value(self.constraint_data.get('最小信噪比', 15))
    
    def reliability_objective(self, params_list: List[Dict]) -> float:
        """
        通信可靠性目标函数
        
        参数:
        params_list: 参数字典列表
        
        返回:
        负的总可靠性（用于最小化）
        """
        total_reliability = 0
        links = self.task_data.get('communication_links', [])
        
        for i, params in enumerate(params_list):
            if i < len(links):
                link = links[i]
                
                # 计算信噪比
                snr = self._calculate_snr(params, link)
                
                # 计算误码率
                ber = self._calculate_bit_error_rate(snr, params.get('modulation', 'BPSK'))
                
                # 假设每个数据包大小为1024比特
                packet_size = link.get('packet_size', 1024)
                
                # 计算包成功率（可靠性）
                reliability = np.power(1 - ber, packet_size)
                
                # 考虑链路重要性
                importance = self._get_link_importance(link)
                
                total_reliability += reliability * importance
        
        # 返回负值用于最小化（原目标是最大化）
        return -total_reliability
    
    def spectral_efficiency_objective(self, params_list: List[Dict]) -> float:
        """
        频谱效率目标函数
        
        参数:
        params_list: 参数字典列表
        
        返回:
        负的总频谱效率（用于最小化）
        """
        total_efficiency = 0
        links = self.task_data.get('communication_links', [])
        
        for i, params in enumerate(params_list):
            if i < len(links):
                link = links[i]
                
                # 获取带宽和信噪比
                bandwidth = params.get('bandwidth', 0)
                snr = self._calculate_snr(params, link)
                
                # Shannon容量
                capacity = bandwidth * np.log2(1 + np.power(10, snr/10))
                
                # 估算数据率（如果没有提供，假设是容量的70%）
                data_rate = link.get('data_rate', capacity * 0.7)
                
                # 计算频谱效率
                spectral_efficiency = data_rate / bandwidth if bandwidth > 0 else 0
                
                total_efficiency += spectral_efficiency
        
        # 返回负值用于最小化（原目标是最大化）
        return -total_efficiency
    
    def energy_efficiency_objective(self, params_list: List[Dict]) -> float:
        """
        能量效率目标函数
        
        参数:
        params_list: 参数字典列表
        
        返回:
        总能量开销
        """
        total_energy = 0
        links = self.task_data.get('communication_links', [])
        
        for i, params in enumerate(params_list):
            if i < len(links):
                link = links[i]
                
                # 获取功率
                power = params.get('power', 0)
                
                # 计算信噪比和误码率
                snr = self._calculate_snr(params, link)
                ber = self._calculate_bit_error_rate(snr, params.get('modulation', 'BPSK'))
                
                # 估算数据率
                bandwidth = params.get('bandwidth', 0)
                capacity = bandwidth * np.log2(1 + np.power(10, snr/10))
                data_rate = link.get('data_rate', capacity * 0.7)
                
                # 通信持续时间（默认为1秒）
                duration = link.get('duration', 1.0)
                
                # 计算成功传输的数据量
                packet_loss_rate = 1 - np.power(1 - ber, 1024)  # 假设1024比特包
                successful_bits = data_rate * duration * (1 - packet_loss_rate)
                
                # 估算电路功耗（通常为发射功率的10-30%）
                circuit_power = 0.2 * power
                
                # 计算总功率
                total_power = power + circuit_power
                
                # 计算每比特能耗
                energy_per_bit = total_power / (successful_bits + 1e-10)  # 避免除零
                
                total_energy += energy_per_bit
        
        return total_energy
    
    def interference_objective(self, params_list: List[Dict]) -> float:
        """
        抗干扰性能目标函数
        
        参数:
        params_list: 参数字典列表
        
        返回:
        负的抗干扰性能（用于最小化）
        """
        interference_metric = 0
        links = self.task_data.get('communication_links', [])
        
        # 第一步：计算链路间干扰
        for i, params_i in enumerate(params_list):
            if i >= len(links):
                continue
                
            # 当前链路的频率
            freq_i = params_i.get('frequency', 0)
            bandwidth_i = params_i.get('bandwidth', 0)
            
            # 检查与其他链路的干扰
            for j, params_j in enumerate(params_list):
                if i == j or j >= len(links):
                    continue
                    
                # 其他链路的频率
                freq_j = params_j.get('frequency', 0)
                bandwidth_j = params_j.get('bandwidth', 0)
                
                # 计算频谱重叠
                overlap = self._calculate_frequency_overlap(
                    freq_i, bandwidth_i, freq_j, bandwidth_j
                )
                
                if overlap > 0:
                    # 存在干扰，降低抗干扰性能
                    interference_metric += overlap
        
        # 第二步：考虑环境干扰
        for i, params in enumerate(params_list):
            if i >= len(links):
                continue
                
            link = links[i]
            
            # 计算环境干扰影响
            env_interference = self._calculate_environmental_interference(
                params.get('frequency', 0),
                params.get('bandwidth', 0),
                link
            )
            
            interference_metric += env_interference
        
        # 第三步：考虑调制方式和极化方式对抗干扰的影响
        for i, params in enumerate(params_list):
            if i >= len(links):
                continue
                
            # 调制方式
            modulation = params.get('modulation', 'BPSK')
            
            # 极化方式
            polarization = params.get('polarization', 'LINEAR')
            
            # 根据调制方式和极化方式计算改进量
            modulation_bonus = self._modulation_interference_resistance(modulation)
            polarization_bonus = self._polarization_interference_resistance(polarization)
            
            # 降低总干扰度量
            interference_metric -= (modulation_bonus + polarization_bonus)
        
        # 取反，使其成为最小化问题
        return interference_metric
    
    def adaptability_objective(self, params_list: List[Dict]) -> float:
        """
        环境适应性目标函数
        
        参数:
        params_list: 参数字典列表
        
        返回:
        负的环境适应性（用于最小化）
        """
        adaptability_score = 0
        links = self.task_data.get('communication_links', [])
        
        for i, params in enumerate(params_list):
            if i >= len(links):
                continue
                
            link = links[i]
            
            # 计算频率适应性
            freq_adaptability = self._calculate_frequency_adaptability(
                params.get('frequency', 0),
                self.sea_state
            )
            
            # 计算功率适应性
            power_adaptability = self._calculate_power_adaptability(
                params.get('power', 0),
                self.emi_intensity
            )
            
            # 计算调制方式适应性
            modulation_adaptability = self._calculate_modulation_adaptability(
                params.get('modulation', 'BPSK'),
                self.sea_state
            )
            
            # 综合评分
            link_adaptability = 0.4 * freq_adaptability + 0.4 * power_adaptability + 0.2 * modulation_adaptability
            
            # 考虑链路重要性
            importance = self._get_link_importance(link)
            
            adaptability_score += link_adaptability * importance
        
        # 取反，使其成为最小化问题
        return -adaptability_score
    
    def _calculate_snr(self, params: Dict, link: Dict) -> float:
        """
        计算信噪比
        
        参数:
        params: 通信参数
        link: 链路信息
        
        返回:
        信噪比（dB）
        """
        # 获取参数
        frequency = params.get('frequency', 0)
        power = params.get('power', 0)
        
        # 从链路信息中获取距离和天线高度
        distance = 100  # 默认100公里
        if 'source_id' in link and 'target_id' in link:
            # 可以根据节点ID计算距离，这里简化为随机值
            distance = link.get('distance', float(abs(link['source_id'] - link['target_id']) * 10))
        
        # 构建链路参数
        link_params = {
            'distance': distance,
            'antenna_height': link.get('antenna_height', 10),  # 默认10米
        }
        
        # 计算噪声
        noise = self.noise_model.calculate_total_noise(frequency, link_params)
        
        # 计算接收功率（考虑路径损耗）
        path_loss = self.noise_model.calculate_propagation_loss(
            frequency, 
            link_params['distance'],
            float(self.env_data.get('depth', 0))
        )
        
        # 发射功率 - 路径损耗 = 接收功率
        rx_power = power - path_loss
        
        # 计算SNR (dB)
        snr = rx_power - noise
        
        return max(snr, 0)  # 确保SNR不为负
    
    def _calculate_bit_error_rate(self, snr: float, modulation: str) -> float:
        """
        计算误码率
        
        参数:
        snr: 信噪比（dB）
        modulation: 调制方式
        
        返回:
        误码率
        """
        # 转换dB到线性
        snr_linear = 10 ** (snr / 10)
        
        # 根据调制方式计算误码率
        if modulation == 'BPSK':
            return 0.5 * np.exp(-snr_linear / 2)
        elif modulation == 'QPSK':
            return 0.5 * np.exp(-snr_linear / 4)
        elif modulation == 'QAM16':
            return 0.2 * np.exp(-snr_linear / 10)
        elif modulation == 'QAM64':
            return 0.1 * np.exp(-snr_linear / 20)
        else:
            # 默认为BPSK
            return 0.5 * np.exp(-snr_linear / 2)
    
    def _get_link_importance(self, link: Dict) -> float:
        """
        计算链路重要性
        
        参数:
        link: 链路信息
        
        返回:
        重要性权重
        """
        # 假设指挥舰船的通信链路最重要
        if self._is_command_ship_link(link):
            return 2.0
        return 1.0
    
    def _is_command_ship_link(self, link: Dict) -> bool:
        """
        判断是否是指挥舰船的通信链路
        
        参数:
        link: 链路信息
        
        返回:
        是否是指挥舰船链路
        """
        if not self.task_data.get('nodes', {}).get('command_ship'):
            return False
            
        command_ship_id = self.task_data['nodes']['command_ship'].get('identity')
        return (link.get('source_id') == command_ship_id or 
                link.get('target_id') == command_ship_id)
    
    def _calculate_frequency_overlap(self, freq1: float, bw1: float, 
                                    freq2: float, bw2: float) -> float:
        """
        计算两个频段的重叠程度
        
        参数:
        freq1, freq2: 中心频率
        bw1, bw2: 带宽
        
        返回:
        重叠度量
        """
        # 计算频段边界
        low1 = freq1 - bw1/2
        high1 = freq1 + bw1/2
        low2 = freq2 - bw2/2
        high2 = freq2 + bw2/2
        
        # 计算重叠部分
        overlap = max(0, min(high1, high2) - max(low1, low2))
        
        # 归一化重叠
        if overlap > 0:
            normalized_overlap = overlap / min(bw1, bw2)
            return normalized_overlap
        return 0
    
    def _calculate_environmental_interference(self, freq: float, 
                                            bandwidth: float, link: Dict) -> float:
        """
        计算环境干扰影响
        
        参数:
        freq: 频率
        bandwidth: 带宽
        link: 链路信息
        
        返回:
        环境干扰度量
        """
        # 计算海况影响
        sea_effect = self.sea_state / 9.0  # 归一化海况
        
        # 计算EMI影响
        emi_effect = self.emi_intensity
        
        # 频率影响（某些频段受干扰更严重）
        freq_effect = 0
        if freq < 500e6:  # HF/VHF频段
            freq_effect = 0.8
        elif freq < 2e9:  # UHF频段
            freq_effect = 0.5
        elif freq < 6e9:  # SHF低频段
            freq_effect = 0.3
        else:  # SHF高频段
            freq_effect = 0.2
        
        # 综合干扰影响
        return 0.4 * sea_effect + 0.4 * emi_effect + 0.2 * freq_effect
    
    def _modulation_interference_resistance(self, modulation: str) -> float:
        """
        计算调制方式的抗干扰能力
        
        参数:
        modulation: 调制方式
        
        返回:
        抗干扰改进量
        """
        resistance_map = {
            'BPSK': 0.8,   # BPSK抗干扰能力最强
            'QPSK': 0.6,
            'QAM16': 0.4,
            'QAM64': 0.2   # 高阶调制抗干扰能力较弱
        }
        return resistance_map.get(modulation, 0.5)
    
    def _polarization_interference_resistance(self, polarization: str) -> float:
        """
        计算极化方式的抗干扰能力
        
        参数:
        polarization: 极化方式
        
        返回:
        抗干扰改进量
        """
        resistance_map = {
            'LINEAR': 0.3,
            'CIRCULAR': 0.5,
            'DUAL': 0.7,
            'ADAPTIVE': 0.9  # 自适应极化抗干扰能力最强
        }
        return resistance_map.get(polarization, 0.4)
    
    def _calculate_frequency_adaptability(self, freq: float, sea_state: float) -> float:
        """
        计算频率对海况的适应性
        
        参数:
        freq: 频率
        sea_state: 海况等级
        
        返回:
        适应性评分（0-1）
        """
        # 不同频段对海况的适应性不同
        if sea_state >= 6:  # 大风浪
            if freq < 500e6:  # HF/VHF适合恶劣海况
                return 0.9
            elif freq < 2e9:
                return 0.7
            elif freq < 6e9:
                return 0.5
            else:
                return 0.3
        else:  # 平静或中等海况
            if freq < 500e6:
                return 0.7
            elif freq < 2e9:
                return 0.8
            elif freq < 6e9:
                return 0.9
            else:
                return 0.7
    
    def _calculate_power_adaptability(self, power: float, emi: float) -> float:
        """
        计算功率对电磁干扰的适应性
        
        参数:
        power: 发射功率
        emi: 电磁干扰强度
        
        返回:
        适应性评分（0-1）
        """
        # 功率与干扰的关系
        power_ratio = power / (100 * emi + 1)  # 归一化
        
        # 适应性评分
        adaptability = min(1.0, max(0.0, 0.2 + 0.8 * (power_ratio / 10)))
        
        return adaptability
    
    def _calculate_modulation_adaptability(self, modulation: str, sea_state: float) -> float:
        """
        计算调制方式对海况的适应性
        
        参数:
        modulation: 调制方式
        sea_state: 海况等级
        
        返回:
        适应性评分（0-1）
        """
        if sea_state >= 6:  # 恶劣海况
            adaptability_map = {
                'BPSK': 0.9,   # 低阶调制更适合恶劣环境
                'QPSK': 0.7,
                'QAM16': 0.4,
                'QAM64': 0.2
            }
        else:  # 良好海况
            adaptability_map = {
                'BPSK': 0.6,
                'QPSK': 0.8,
                'QAM16': 0.9,
                'QAM64': 0.7
            }
        
        return adaptability_map.get(modulation, 0.5)

    def _parse_numeric_value(self, value):
        """从可能包含单位的字符串中提取数值"""
        if value is None:
            return 0
            
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # 移除所有非数字、非小数点、非正负号的字符
            numeric_chars = ''.join(c for c in value if c.isdigit() or c == '.' or c == '-')
            if numeric_chars:
                try:
                    return float(numeric_chars)
                except ValueError:
                    pass
        
        # 默认返回0
        return 0.0