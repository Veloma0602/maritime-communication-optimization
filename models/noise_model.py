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
    
    def calculate_propagation_loss(self, freq, distance, depth):
        """计算传播损耗"""
        # 几何扩展损耗
        spreading_loss = 20 * np.log10(distance)
        
        # 吸收损耗
        alpha = (0.11 * freq**2)/(1 + freq**2) + \
               (44 * freq**2)/(4100 + freq**2) + \
               (3e-4 * freq**2)
        absorption_loss = alpha * distance/1000
        
        # 深度影响
        depth_factor = np.exp(-depth/1000)
        
        return spreading_loss + absorption_loss * depth_factor
    
    def calculate_total_noise(self, freq, link_params):
        """计算总噪声"""
        N_amb = self.calculate_ambient_noise(freq)
        N_mp = self.calculate_multipath_noise(
            freq,
            link_params['antenna_height'],
            link_params['distance']
        )
        L_prop = self.calculate_propagation_loss(
            freq,
            link_params['distance'],
            float(self.env_data['depth'])
        )
        
        # 人为干扰
        N_int = 10 * np.log10(float(self.env_data['electromagnetic_intensity']))
        
        return N_amb + N_mp + L_prop + N_int
    
    def _process_env_data(self):
        """处理环境数据，提取噪声计算所需的参数"""
        # 海况等级
        self.sea_state = self._parse_numeric_value(self.env_data.get('海况等级', 3))
        
        # 降雨率（如果有的话）
        self.rain_rate = self._parse_numeric_value(self.env_data.get('降雨率', 0))
        
        # 电磁干扰强度
        self.emi_level = self._parse_numeric_value(self.env_data.get('电磁干扰强度', 0.5))
        
        # 背景噪声 (如 "-107dBm")
        self.background_noise = self._parse_numeric_value(self.env_data.get('背景噪声', -100))
        
        # 多径效应
        self.multipath_effect = self._parse_numeric_value(self.env_data.get('多径效应', 0.3))
        
        # 海水温度和盐度
        self.temperature = self._parse_numeric_value(self.env_data.get('温度', 20))
        self.salinity = self._parse_numeric_value(self.env_data.get('盐度', 35))

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