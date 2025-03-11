from typing import Dict, Any
import numpy as np


class NoiseModel:
    def __init__(self, env_data):
        self.env_data = env_data
        
    def calculate_ambient_noise(self, freq):
        """计算环境噪声"""
        sea_state = float(self.env_data['sea_state'])
        rain_rate = float(self.env_data.get('rain_rate', 0))
        
        # 海浪噪声
        N_s = 40 + 20 * (sea_state/9) + 26 * np.log10(freq/1e3)
        
        # 雨噪声
        N_r = 15 + 10 * np.log10(rain_rate + 1e-10) + 10 * np.log10(freq/1e3)
        
        # 热噪声
        N_th = -15 + 20 * np.log10(freq/1e3)
        
        return 10 * np.log10(10**(N_s/10) + 10**(N_r/10) + 10**(N_th/10))
    
    def calculate_multipath_noise(self, freq, height, distance):
        """计算多径噪声"""
        wavelength = 3e8 / freq
        path_diff = np.sqrt(distance**2 + 4*height**2) - distance
        phase_diff = 2 * np.pi * path_diff / wavelength
        return 20 * np.log10(abs(1 + np.exp(1j*phase_diff)))
    
    def calculate_propagation_loss(self, freq: float, distance: float, depth: float = 0) -> float:
        """
        计算传播损耗
        
        参数:
        freq: 频率 (Hz)
        distance: 传播距离 (km)
        depth: 水深 (m)，默认为0表示海面
        
        返回:
        传播损耗 (dB)
        """
        # 确保深度是数值
        if not isinstance(depth, (int, float)):
            try:
                depth = float(depth)
            except (ValueError, TypeError):
                depth = 0
        
        # 几何扩展损耗
        # 自由空间损耗 = 20*log10(4*pi*d/λ)
        spreading_loss = 20 * np.log10(max(0.1, distance)) + 20 * np.log10(max(1e6, freq)) - 147.56
        
        # 计算吸收系数 (dB/km)
        # 这是一个简化的模型，实际海水吸收与温度、盐度、深度和频率相关
        freq_MHz = freq / 1e6
        
        # 简化的吸收系数计算
        alpha = (0.11 * freq_MHz**2) / (1 + freq_MHz**2) + \
            (44 * freq_MHz**2) / (4100 + freq_MHz**2) + \
            (3e-4 * freq_MHz**2)
        
        # 吸收损耗
        absorption_loss = alpha * distance
        
        # 深度影响因子
        # 实际上，随着深度增加，声道效应会影响传播，这里简化处理
        depth_factor = np.exp(-depth/1000) if depth > 0 else 1.0
        
        # 总传播损耗 (dB)
        total_loss = spreading_loss + absorption_loss * depth_factor
        
        return total_loss
    
    def calculate_total_noise(self, freq: float, link_params: Dict) -> float:
        """
        计算总噪声
        
        参数:
        freq: 频率 (Hz)
        link_params: 链路参数
        
        返回:
        总噪声 (dB)
        """
        # 提取链路参数
        antenna_height = link_params.get('antenna_height', 10)  # 默认10米
        distance = link_params.get('distance', 100)  # 默认100公里
        
        # 安全获取深度值，提供默认值
        depth = 0
        if 'depth' in self.env_data:
            depth = self._parse_numeric_value(self.env_data['depth'])
        
        # 计算环境噪声
        N_amb = self.calculate_ambient_noise(freq)
        
        # 计算多径噪声
        N_mp = self.calculate_multipath_noise(freq, antenna_height, distance)
        
        # 计算传播损耗
        L_prop = self.calculate_propagation_loss(freq, distance, depth)
        
        # 人为电磁干扰
        # 干扰强度转换为dB
        emi_value = getattr(self, 'emi_intensity', 0.5)  # 如果不存在，使用默认值0.5
        N_int = 10 * np.log10(emi_value + 1e-10)
        
        # 总噪声
        total_noise = N_amb + N_mp + L_prop + N_int
        
        return total_noise
    
    def _process_env_data(self):
        """处理环境数据，提取噪声计算所需的参数"""
        # 海况等级 - 同时支持中文和英文键名
        self.sea_state = self._parse_numeric_value(
            self.env_data.get('海况等级', self.env_data.get('sea_state', 3))
        )
        
        # 降雨率
        self.rain_rate = self._parse_numeric_value(
            self.env_data.get('降雨率', self.env_data.get('rain_rate', 0))
        )
        
        # 电磁干扰强度
        self.emi_level = self._parse_numeric_value(
            self.env_data.get('电磁干扰强度', self.env_data.get('emi_intensity', 0.5))
        )
        # 为兼容性添加别名
        self.emi_level = self.emi_intensity
        
        # 背景噪声
        self.background_noise = self._parse_numeric_value(
            self.env_data.get('背景噪声', self.env_data.get('background_noise', -100))
        )
        
        # 多径效应
        self.multipath_effect = self._parse_numeric_value(
            self.env_data.get('多径效应', self.env_data.get('multipath_effect', 0.3))
        )
        
        # 海水温度和盐度
        self.temperature = self._parse_numeric_value(
            self.env_data.get('温度', self.env_data.get('temperature', 20))
        )
        self.salinity = self._parse_numeric_value(
            self.env_data.get('盐度', self.env_data.get('salinity', 35))
        )
        
        # 调试输出
        print(f"环境数据处理: 海况={self.sea_state}, 干扰={self.emi_level}")

    def _parse_numeric_value(self, value):
        """从可能包含单位的字符串中提取数值"""
        if value is None:
            return 0
            
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # 尝试直接转换
            try:
                return float(value)
            except ValueError:
                pass
                
            # 提取数字部分 (包括负号和小数点)
            import re
            numeric_match = re.search(r'-?\d+\.?\d*', value)
            if numeric_match:
                try:
                    return float(numeric_match.group())
                except ValueError:
                    pass
        
        # 默认返回0
        return 0.0