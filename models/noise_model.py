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