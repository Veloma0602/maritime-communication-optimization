import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any, Tuple  

import matplotlib.font_manager as fm
import matplotlib as mpl

# 查找字体文件并手动添加
font_path = '/root/maritime-communication-optimization/wqy-microhei/wqy-microhei.ttc'  
if os.path.exists(font_path):
    fontprop = fm.FontProperties(fname=font_path)
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] + mpl.rcParams['font.sans-serif']
    print(f"成功加载中文字体: {font_path}")
else:
    print(f"找不到中文字体文件: {font_path}")
    # 回退到英文标签

# 定义中英文标签映射为全局变量
LABEL_MAP = {
    "通信可靠性": "Reliability",
    "频谱效率": "Spectral Efficiency",
    "能量效率": "Energy Efficiency",
    "抗干扰性能": "Interference",
    "环境适应性": "Adaptability",
    "约束违反度": "Constraint Violation",
    "代数": "Generation",
    "适应度": "Fitness",
    "最优适应度": "Best Fitness",
    "平均适应度": "Average Fitness",
    "最小约束违反度": "Min Constraint Violation",
    "平均约束违反度": "Avg Constraint Violation",
    "频率": "Frequency",
    "带宽": "Bandwidth",
    "功率": "Power",
    "调制方式": "Modulation",
    "极化方式": "Polarization"
}

# 全局函数用于替换中文标签
def translate_label(label):
    for cn, en in LABEL_MAP.items():
        label = label.replace(cn, en)
    return label



class OptimizationVisualizer:
    """优化结果可视化工具，用于分析NSGA-II多目标优化的结果"""
    # 在visualization.py中
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图表风格
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 完全使用英文标签
        self.objective_names = [
            "Reliability (Min Better)", 
            "Spectral Efficiency (Min Better)", 
            "Energy Efficiency (Max Better)", 
            "Interference (Min Better)", 
            "Adaptability (Min Better)"
        ]
        
        self.fitness_names = [
            "Reliability (Max Better)", 
            "Spectral Efficiency (Max Better)", 
            "Energy Efficiency (Min Better)", 
            "Interference (Max Better)", 
            "Adaptability (Max Better)"
        ]
        
        # 参数名称英文化
        self.param_names = {
            '频率': 'Frequency',
            '带宽': 'Bandwidth',
            '功率': 'Power',
            '调制方式': 'Modulation',
            '极化方式': 'Polarization'
        }
        
    def visualize_objectives(self, objectives: np.ndarray, task_id: str):
        """
        可视化目标函数值
        
        参数:
        objectives: 目标函数值数组，形状为 (n_solutions, n_objectives)
        task_id: 任务ID，用于标题和文件名
        """
        n_solutions, n_objectives = objectives.shape
        
        # 目标函数箱线图
        plt.figure(figsize=(12, 8))
        plt.boxplot(objectives, labels=self.objective_names)
        plt.title(translate_label(f'任务 {task_id} 的目标函数分布'))
        plt.ylabel('目标函数值')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{task_id}_objectives_boxplot.png'))
        plt.close()
        
        # 转换目标函数：由于NSGA-II是最小化问题，我们需要对部分目标取反
        converted_objectives = np.copy(objectives)
        # 通信可靠性、频谱效率、抗干扰性能和环境适应性是取反的
        converted_objectives[:, 0] *= -1  # 通信可靠性
        converted_objectives[:, 1] *= -1  # 频谱效率
        converted_objectives[:, 3] *= -1  # 抗干扰性能
        converted_objectives[:, 4] *= -1  # 环境适应性
        
        # 实际适应度箱线图（转换后的目标）
        plt.figure(figsize=(12, 8))
        plt.boxplot(converted_objectives, labels=self.fitness_names)
        plt.title(translate_label(f'任务 {task_id} 的实际适应度分布（越大越好）'))
        plt.ylabel('适应度值')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{task_id}_fitness_boxplot.png'))
        plt.close()
        
        # 如果解的数量足够，绘制散点矩阵
        if n_solutions > 1:
            try:
                import pandas as pd
                import seaborn as sns
                
                # 创建DataFrame
                obj_df = pd.DataFrame(converted_objectives, columns=self.fitness_names)
                
                # 绘制散点矩阵
                plt.figure(figsize=(15, 15))
                sns.pairplot(obj_df)
                plt.suptitle(f'任务 {task_id} 的目标函数散点矩阵', y=1.02)
                plt.savefig(os.path.join(self.output_dir, f'{task_id}_objectives_pairplot.png'))
                plt.close()
            except (ImportError, Exception) as e:
                print(f"生成散点矩阵时出错: {str(e)}")
    
    def visualize_parameter_distribution(self, variables: np.ndarray, task_id: str, n_links: int):
        """
        可视化参数分布
        
        参数:
        variables: 参数数组，形状为 (n_solutions, n_parameters)
        task_id: 任务ID
        n_links: 通信链路数量
        """
        n_solutions, n_parameters = variables.shape
        
        # 参数名称
        param_groups = ['频率', '带宽', '功率', '调制方式', '极化方式']
        
        # 重塑参数数组
        reshaped_vars = []
        for solution in variables:
            # 确保解向量长度足够
            if len(solution) >= n_links * 5:
                # 取前n_links*5个元素，重塑为(n_links, 5)
                solution_reshaped = solution[:n_links*5].reshape(n_links, 5)
                reshaped_vars.append(solution_reshaped)
        
        if not reshaped_vars:
            print("没有足够长度的解向量，无法可视化参数分布")
            return
            
        reshaped_vars = np.array(reshaped_vars)
        
        # 为每种参数创建箱线图
        for param_idx, param_name in enumerate(param_groups):
            plt.figure(figsize=(10, 6))
            
            # 获取所有链路的该类参数
            param_values = reshaped_vars[:, :, param_idx]
            
            # 为每个链路创建标签
            labels = [f'链路{i+1}' for i in range(n_links)]
            
            # 绘制箱线图
            plt.boxplot(param_values, labels=labels)
            plt.title(translate_label(f'任务 {task_id} 的 {param_name} 参数分布'))
            plt.ylabel(translate_label(param_name))
            plt.xlabel('通信链路')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{task_id}_param_{param_name}.png'))
            plt.close()
    
    def visualize_convergence(self, history: Dict, task_id: str):
        """
        可视化收敛曲线
        
        参数:
        history: 历史记录，包含每代的评估数据
        task_id: 任务ID
        """
        if not history or 'n_gen' not in history or 'cv_min' not in history:
            print("历史记录不完整，无法绘制收敛曲线")
            return
            
        generations = history['n_gen']
        cv_min = history['cv_min']
        cv_avg = history['cv_avg']
        
        # 绘制约束违反度曲线
        plt.figure(figsize=(12, 6))
        plt.plot(generations, cv_min, label='最小约束违反度', marker='o')
        plt.plot(generations, cv_avg, label='平均约束违反度', marker='x')
        plt.xlabel('代数')
        plt.ylabel('约束违反度')
        plt.title(translate_label(f'任务 {task_id} 优化过程中的约束违反度变化'))
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # 对数坐标，方便观察大值
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{task_id}_constraint_violation.png'))
        plt.close()
        
        # 修改visualization.py中的问题代码
        if 'f_avg' in history and 'f_min' in history:
            f_avg = history['f_avg']
            f_min = history['f_min']
            
            # 绘制适应度曲线 - 为每个目标函数单独绘制
            fig, axes = plt.subplots(len(f_min), 1, figsize=(12, 4 * len(f_min)), sharex=True)
            
            for i, (f_min_obj, f_avg_obj) in enumerate(zip(f_min, f_avg)):
                ax = axes[i] if len(f_min) > 1 else axes
                ax.plot(generations, f_min_obj, label='最优适应度', marker='o')
                ax.plot(generations, f_avg_obj, label='平均适应度', marker='x')
                ax.set_xlabel('代数')
                ax.set_ylabel(f'目标 {i+1} 适应度')
                ax.set_title(f'目标 {i+1} 适应度变化')
                ax.legend()
                ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{task_id}_fitness.png'))
            plt.close()
    
    def save_optimization_results(self, results: List[Dict], task_id: str):
        """
        保存优化结果为文本报告
        
        参数:
        results: 优化结果字典列表
        task_id: 任务ID
        """
        output_file = os.path.join(self.output_dir, f'{task_id}_optimization_report.txt')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== 任务 {task_id} 优化结果报告 ===\n\n")
            f.write(f"找到 {len(results)} 个非支配解\n\n")
            
            for i, result in enumerate(results):
                f.write(f"解 {i+1}:\n")
                f.write(f"  加权得分: {result.get('weighted_objective', 'N/A')}\n")
                
                # 写入目标函数值
                objectives = result.get('objectives', {})
                f.write("  目标函数值:\n")
                for name, value in objectives.items():
                    f.write(f"    {name}: {value}\n")
                
                # 写入参数
                parameters = result.get('parameters', [])
                f.write("  参数配置:\n")
                for j, params in enumerate(parameters):
                    f.write(f"    链路 {j+1}:\n")
                    for param_name, param_value in params.items():
                        f.write(f"      {param_name}: {param_value}\n")
                f.write("\n")
            
            f.write("=== 报告结束 ===\n")
        
        print(f"优化结果报告已保存到 {output_file}")
        
    def print_summary(self, results: List[Dict], task_id: str):
        """
        打印结果摘要
        
        参数:
        results: 优化结果字典列表
        task_id: 任务ID
        """
        if not results:
            print(f"任务 {task_id} 没有找到有效的优化结果")
            return
            
        print(f"\n=== 任务 {task_id} 优化结果摘要 ===")
        print(f"找到 {len(results)} 个非支配解")
        
        if len(results) > 0:
            best_result = results[0]  # 假设结果已按加权目标排序
            print(f"\n最优解:")
            print(f"  加权得分: {best_result.get('weighted_objective', 'N/A')}")
            
            # 打印目标函数值
            objectives = best_result.get('objectives', {})
            print("  目标函数值:")
            for name, value in objectives.items():
                print(f"    {name}: {value}")
            
            # 打印部分参数样例
            parameters = best_result.get('parameters', [])
            print(f"  参数配置样例 (共 {len(parameters)} 个链路):")
            for j, params in enumerate(parameters[:2]):  # 只打印前两个链路的参数
                print(f"    链路 {j+1}:")
                for param_name, param_value in params.items():
                    print(f"      {param_name}: {param_value}")
            if len(parameters) > 2:
                print(f"    (更多链路参数请查看完整报告...)")
                
        print("\n完整结果已保存到报告文件")