import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any, Tuple

class OptimizationVisualizer:
    """优化结果可视化工具，用于分析NSGA-II多目标优化的结果"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图表风格
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            # 如果seaborn样式不可用，使用基本样式
            pass
        
        # 使用英文标签，但中文注释
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
        
        # 参数名称
        self.param_names = {
            'Frequency': 'Frequency',
            'Bandwidth': 'Bandwidth',
            'Power': 'Power',
            'Modulation': 'Modulation',
            'Polarization': 'Polarization'
        }
        
    def visualize_objectives(self, objectives: np.ndarray, task_id: str):
        """
        可视化目标函数值
        
        参数:
        objectives: 目标函数值数组，形状为 (n_solutions, n_objectives)
        task_id: 任务ID，用于标题和文件名
        """
        try:
            n_solutions, n_objectives = objectives.shape
            
            # 目标函数箱线图
            plt.figure(figsize=(12, 8))
            plt.boxplot(objectives, labels=self.objective_names)
            plt.title(f'Objective Function Distribution for Task {task_id}')
            plt.ylabel('Objective Value')
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
            plt.title(f'Actual Fitness Distribution for Task {task_id} (Higher is Better)')
            plt.ylabel('Fitness Value')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{task_id}_fitness_boxplot.png'))
            plt.close()
            
            # 如果解的数量足够，绘制散点矩阵
            if n_solutions > 1:
                try:
                    import pandas as pd
                    
                    # 尝试导入seaborn
                    try:
                        import seaborn as sns
                        
                        # 创建DataFrame
                        obj_df = pd.DataFrame(converted_objectives, columns=self.fitness_names)
                        
                        # 绘制散点矩阵
                        plt.figure(figsize=(15, 15))
                        sns.pairplot(obj_df)
                        plt.suptitle(f'Objective Function Scatter Matrix for Task {task_id}', y=1.02)
                        plt.savefig(os.path.join(self.output_dir, f'{task_id}_objectives_pairplot.png'))
                        plt.close()
                    except ImportError:
                        print("未安装seaborn库，跳过散点矩阵绘制。可以使用 'pip install seaborn' 安装。")
                        
                        # 如果seaborn不可用，使用pandas的散点矩阵功能
                        try:
                            obj_df = pd.DataFrame(converted_objectives, columns=self.fitness_names)
                            scatter_matrix = pd.plotting.scatter_matrix(obj_df, figsize=(15, 15))
                            plt.suptitle(f'Objective Function Scatter Matrix for Task {task_id}', y=1.02)
                            plt.savefig(os.path.join(self.output_dir, f'{task_id}_objectives_pairplot.png'))
                            plt.close()
                        except:
                            print("无法使用pandas的散点矩阵功能，跳过散点矩阵绘制。")
                except ImportError:
                    print("未安装pandas库，跳过散点矩阵绘制。可以使用 'pip install pandas' 安装。")
                    
        except Exception as e:
            print(f"可视化目标函数时出错: {str(e)}")
    
    def visualize_parameter_distribution(self, variables: np.ndarray, task_id: str, n_links: int):
        """
        可视化参数分布
        
        参数:
        variables: 参数数组，形状为 (n_solutions, n_parameters)
        task_id: 任务ID
        n_links: 通信链路数量
        """
        try:
            n_solutions, n_parameters = variables.shape
            
            # 参数名称
            param_groups = ['Frequency', 'Bandwidth', 'Power', 'Modulation', 'Polarization']
            
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
                labels = [f'Link {i+1}' for i in range(n_links)]
                
                # 绘制箱线图
                plt.boxplot(param_values, labels=labels)
                plt.title(f'{param_name} Parameter Distribution for Task {task_id}')
                plt.ylabel(param_name)
                plt.xlabel('Communication Link')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{task_id}_param_{param_name}.png'))
                plt.close()
                
        except Exception as e:
            print(f"可视化参数分布时出错: {str(e)}")
    
    def visualize_convergence(self, history: Dict, task_id: str):
        """
        可视化收敛曲线
        
        参数:
        history: 历史记录，包含每代的评估数据
        task_id: 任务ID
        """
        try:
            if not history or 'n_gen' not in history or 'cv_min' not in history:
                print("历史记录不完整，无法绘制收敛曲线")
                return
                
            generations = history['n_gen']
            cv_min = history['cv_min']
            cv_avg = history['cv_avg']
            
            # 绘制约束违反度曲线
            plt.figure(figsize=(12, 6))
            plt.plot(generations, cv_min, label='Min Constraint Violation', marker='o')
            plt.plot(generations, cv_avg, label='Avg Constraint Violation', marker='x')
            plt.xlabel('Generation')
            plt.ylabel('Constraint Violation')
            plt.title(f'Constraint Violation Changes During Optimization for Task {task_id}')
            plt.legend()
            plt.grid(True)
            
            # 只有在有正值时才使用对数缩放
            if any(val > 0 for val in cv_min) or any(val > 0 for val in cv_avg):
                try:
                    plt.yscale('log')  # 对数坐标，方便观察大值
                except Exception as e:
                    print(f"设置对数坐标失败: {str(e)}")
                    
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{task_id}_constraint_violation.png'))
            plt.close()
            
            if 'f_avg' in history and 'f_min' in history:
                f_avg = history['f_avg']
                f_min = history['f_min']
                
                # 绘制适应度曲线 - 为每个目标函数单独绘制
                if f_min and len(f_min) > 0:
                    fig, axes = plt.subplots(len(f_min), 1, figsize=(12, 4 * len(f_min)), sharex=True)
                    
                    for i, (f_min_obj, f_avg_obj) in enumerate(zip(f_min, f_avg)):
                        ax = axes[i] if len(f_min) > 1 else axes
                        ax.plot(generations, f_min_obj, label='Best Fitness', marker='o')
                        ax.plot(generations, f_avg_obj, label='Average Fitness', marker='x')
                        ax.set_xlabel('Generation')
                        ax.set_ylabel(f'Objective {i+1} Fitness')
                        ax.set_title(f'Objective {i+1} Fitness Changes')
                        ax.legend()
                        ax.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, f'{task_id}_fitness.png'))
                    plt.close()
                    
        except Exception as e:
            print(f"可视化收敛曲线时出错: {str(e)}")
    
    def save_optimization_results(self, results: List[Dict], task_id: str):
        """
        保存优化结果为文本报告
        
        参数:
        results: 优化结果字典列表
        task_id: 任务ID
        """
        try:
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
            
        except Exception as e:
            print(f"保存优化结果时出错: {str(e)}")
        
    def print_summary(self, results: List[Dict], task_id: str):
        """
        打印结果摘要
        
        参数:
        results: 优化结果字典列表
        task_id: 任务ID
        """
        try:
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
            
        except Exception as e:
            print(f"打印摘要时出错: {str(e)}")