from optimization.nsga2_optimizer import CommunicationOptimizer
from data.neo4j_handler import Neo4jHandler
from config.parameters import OptimizationConfig
import numpy as np

def main():
    # 初始化Neo4j处理器
    neo4j_handler = Neo4jHandler(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password="your_password"
    )
    
    # 获取任务数据
    task_id = "rw001"
    task_data = neo4j_handler.get_task_data(task_id)
    env_data = neo4j_handler.get_environment_data(task_id)
    constraint_data = neo4j_handler.get_constraint_data(task_id)
    
    # 初始化优化器
    optimizer = CommunicationOptimizer(
        task_data=task_data,
        env_data=env_data,
        constraint_data=constraint_data,
        config=OptimizationConfig()
    )
    
    # 运行优化
    pareto_front, optimal_variables = optimizer.optimize()
    
    # 保存结果
    neo4j_handler.save_optimization_results(
        task_id=task_id,
        pareto_front=pareto_front,
        optimal_variables=optimal_variables
    )

if __name__ == "__main__":
    main()