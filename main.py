from optimization.nsga2_optimizer import CommunicationOptimizer
from data.neo4j_handler import Neo4jHandler
from config.parameters import OptimizationConfig
import numpy as np
import logging
import argparse
import json
import time
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """主程序入口，运行通信参数优化"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='海上通信参数多目标优化系统')
    parser.add_argument('--task-id', type=str, default="rw001", help='要优化的任务ID')
    parser.add_argument('--db-uri', type=str, default="neo4j://47.98.246.127:7687", help='Neo4j数据库URI')
    parser.add_argument('--db-user', type=str, default="neo4j", help='Neo4j用户名')
    parser.add_argument('--db-password', type=str, default="12345678", help='Neo4j密码')
    parser.add_argument('--output-dir', type=str, default="results", help='结果输出目录')
    parser.add_argument('--generations', type=int, default=None, help='NSGA-II迭代代数')
    parser.add_argument('--population', type=int, default=None, help='种群大小')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
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
        pareto_front, optimal_variables = optimizer.optimize()
        
        # 处理优化结果
        logger.info(f"优化完成，得到 {len(pareto_front)} 个非支配解")
        results = optimizer.post_process_solutions(pareto_front, optimal_variables)
        
        # 保存结果到Neo4j
        logger.info(f"保存优化结果到Neo4j数据库")
        neo4j_handler.save_optimization_results(
            task_id=args.task_id,
            pareto_front=pareto_front,
            optimal_variables=optimal_variables
        )
        
        # 保存详细结果到JSON文件
        result_file = os.path.join(args.output_dir, f"{args.task_id}_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'task_id': args.task_id,
                'optimization_config': {
                    'population_size': config.population_size,
                    'n_generations': config.n_generations,
                    'mutation_prob': config.mutation_prob,
                    'crossover_prob': config.crossover_prob
                },
                'results': results,
                'execution_time': time.time() - start_time
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细结果已保存到 {result_file}")
        
        # 关闭Neo4j连接
        neo4j_handler.close()
        
        # 输出执行时间
        elapsed_time = time.time() - start_time
        logger.info(f"优化完成，耗时: {elapsed_time:.2f}秒")
        
        # 输出最优解信息
        if results:
            best_solution = results[0]  # 根据加权目标排序后的第一个解
            logger.info(f"最优解指标:")
            logger.info(f"  可靠性: {best_solution['objectives']['reliability']:.4f}")
            logger.info(f"  频谱效率: {best_solution['objectives']['spectral_efficiency']:.4f}")
            logger.info(f"  能量效率: {best_solution['objectives']['energy_efficiency']:.4f}")
            logger.info(f"  抗干扰性: {best_solution['objectives']['interference']:.4f}")
            logger.info(f"  环境适应性: {best_solution['objectives']['adaptability']:.4f}")
            logger.info(f"  加权评分: {best_solution['weighted_objective']:.4f}")
    
    except Exception as e:
        logger.error(f"优化过程中发生错误: {str(e)}", exc_info=True)
    
    logger.info("=== 优化任务结束 ===")

if __name__ == "__main__":
    main()