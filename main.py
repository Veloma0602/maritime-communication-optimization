from optimization.nsga2_optimizer import CommunicationOptimizer
from data.neo4j_handler import Neo4jHandler
from config.parameters import OptimizationConfig
from optimization.visualization import OptimizationVisualizer
from utils.logging_config import setup_logging  # 导入新的日志设置
import numpy as np
import argparse
import json
import time
import os
import traceback

def main():
    """主程序入口，运行通信参数优化"""
    # 设置带时间戳文件名的日志系统
    logger = setup_logging()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='海上通信参数多目标优化系统')
    parser.add_argument('--task-id', type=str, default="rw003", help='要优化的任务ID')
    parser.add_argument('--db-uri', type=str, default="bolt://localhost:7687", help='Neo4j数据库URI')
    parser.add_argument('--db-user', type=str, default="neo4j", help='Neo4j用户名')
    parser.add_argument('--db-password', type=str, default="12345678", help='Neo4j密码')
    parser.add_argument('--output-dir', type=str, default="results", help='结果输出目录')
    parser.add_argument('--generations', type=int, default=None, help='NSGA-II迭代代数')
    parser.add_argument('--population', type=int, default=None, help='种群大小')
    parser.add_argument('--no-save', action='store_true', help='不保存结果到Neo4j')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化可视化工具
    visualizer = OptimizationVisualizer(args.output_dir)
    
    # 记录开始时间
    start_time = time.time()
    logger.info(f"=== 开始优化任务 {args.task_id} ===")
    
    try:
        # 初始化Neo4j处理器
        logger.info(f"连接到Neo4j数据库 {args.db_uri}")
        neo4j_handler = Neo4jHandler(
            uri=args.db_uri,
            user=args.db_user,
            password=args.db_password
        )
        
        # 执行测试查询
        logger.info("执行测试查询...")
        neo4j_handler.test_query(args.task_id)
        
        # 获取任务数据
        logger.info(f"获取任务 {args.task_id} 的数据")
        task_data = neo4j_handler.get_task_data(args.task_id)
        if not task_data:
            logger.error(f"无法获取任务 {args.task_id} 的数据")
            return
        
        # 获取环境数据
        logger.info(f"获取任务 {args.task_id} 的环境数据")
        env_data = neo4j_handler.get_environment_data(args.task_id)
        if not env_data:
            logger.error(f"无法获取任务 {args.task_id} 的环境数据")
            return
        
        # 获取约束数据
        logger.info(f"获取任务 {args.task_id} 的约束数据")
        constraint_data = neo4j_handler.get_constraint_data(args.task_id)
        if not constraint_data:
            logger.error(f"无法获取任务 {args.task_id} 的约束数据")
            return
        
        # 初始化优化配置
        config = OptimizationConfig()
        
        # 如果命令行参数中指定了种群大小或迭代代数，则覆盖默认配置
        if args.generations:
            config.n_generations = args.generations
        if args.population:
            config.population_size = args.population
        
        # 输出优化配置信息
        logger.info(f"优化配置: 种群大小={config.population_size}, 迭代代数={config.n_generations}")
        
        # 初始化优化器
        logger.info("初始化优化器")
        optimizer = CommunicationOptimizer(
            task_data=task_data,
            env_data=env_data,
            constraint_data=constraint_data,
            config=config,
            neo4j_handler=neo4j_handler
        )
        
        # 运行优化
        logger.info("开始运行NSGA-II优化")
        pareto_front, optimal_variables, history = optimizer.optimize()
        
        # 处理优化结果
        n_solutions = len(pareto_front)
        logger.info(f"优化完成，得到 {n_solutions} 个非支配解")
        
        # 后处理结果
        results = optimizer.post_process_solutions(pareto_front, optimal_variables)
        
        # 可视化结果
        logger.info("生成结果可视化...")
        
        # 可视化目标函数分布
        visualizer.visualize_objectives(pareto_front, args.task_id)
        
        # 可视化参数分布
        visualizer.visualize_parameter_distribution(
            optimal_variables, 
            args.task_id, 
            optimizer.n_links
        )
        
        # 可视化收敛曲线
        visualizer.visualize_convergence(history, args.task_id)
        
        # 保存结果报告
        visualizer.save_optimization_results(results, args.task_id)
        
        # 打印结果摘要
        visualizer.print_summary(results, args.task_id)
        
        # 如果不禁用保存，则保存到Neo4j
        if not args.no_save:
            logger.info(f"保存优化结果到Neo4j数据库")
            neo4j_handler.save_optimization_results(
                task_id=args.task_id,
                pareto_front=pareto_front,
                optimal_variables=optimal_variables
            )
        else:
            logger.info("已禁用保存到Neo4j")
        
        # 保存详细结果到JSON文件
        result_file = os.path.join(args.output_dir, f"{args.task_id}_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'task_id': args.task_id,
                'optimization_config': config.to_dict(),
                'results': [
                    {
                        'objectives': r['objectives'],
                        'weighted_objective': r['weighted_objective'],
                        # 不保存参数以减小文件大小
                    } for r in results
                ],
                'execution_time': time.time() - start_time
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细结果已保存到 {result_file}")
        
        # 关闭Neo4j连接
        neo4j_handler.close()
        
        # 输出执行时间
        elapsed_time = time.time() - start_time
        logger.info(f"优化完成，耗时: {elapsed_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"优化过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())
    
    logger.info("=== 优化任务结束 ===")

if __name__ == "__main__":
    main()