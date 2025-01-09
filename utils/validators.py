import numpy as np
from typing import Dict, Any

class DataValidator:
    @staticmethod
    def validate_task_data(data: Dict) -> bool:
        """验证任务数据完整性"""
        required_fields = ['任务编号', '任务区域', '任务时间范围', '兵力组成']
        return all(field in data for field in required_fields)
    
    @staticmethod
    def validate_environment_data(data: Dict) -> bool:
        """验证环境数据完整性"""
        required_fields = ['海况等级', '电磁干扰强度', '地理特征']
        return all(field in data for field in required_fields)
    
    @staticmethod
    def validate_constraint_data(data: Dict) -> bool:
        """验证约束条件数据完整性"""
        required_fields = [
            '频谱最小频率', '频谱最大频率',
            '最小可靠性要求', '最大时延要求'
        ]
        return all(field in data for field in required_fields)
    
    @staticmethod
    def validate_solution(solution: np.ndarray,
                         config: Dict[str, Any]) -> bool:
        """验证解的有效性"""
        try:
            # 基本形状检查
            if len(solution) % 5 != 0:
                return False
            
            # 参数范围检查
            frequencies = solution[::5]
            bandwidths = solution[1::5]
            powers = solution[2::5]
            
            # 频率约束
            freq_valid = np.all(
                (frequencies >= config['freq_min']) &
                (frequencies <= config['freq_max'])
            )
            if not freq_valid:
                return False
            
            # 功率约束
            power_valid = np.all(
                (powers >= config['power_min']) &
                (powers <= config['power_max'])
            )
            if not power_valid:
                return False
            
            # 带宽约束
            bandwidth_valid = np.all(
                (bandwidths >= config['bandwidth_min']) &
                (bandwidths <= config['bandwidth_max'])
            )
            if not bandwidth_valid:
                return False
            
            return True
            
        except Exception as e:
            print(f"Solution validation error: {str(e)}")
            return False