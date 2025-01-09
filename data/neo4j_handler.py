from neo4j import GraphDatabase
from typing import Dict, List
import numpy as np

class Neo4jHandler:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        """关闭数据库连接"""
        self.driver.close()
        
    def get_task_data(self, task_id: str) -> Dict:
        """获取任务数据"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:任务 {任务编号: $task_id})
                OPTIONAL MATCH (t)-[:部署单位]->(n:节点)
                OPTIONAL MATCH (s:节点)-[r:通信手段]->(d:节点)
                WHERE s IN collect(n) AND d IN collect(n)
                RETURN t, collect(n) as nodes, collect(r) as relationships
                """, task_id=task_id)
            record = result.single()
            if record:
                return self.process_task_record(record)
            return None
            
    def get_environment_data(self, task_id: str) -> Dict:
        """获取环境条件数据"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:任务 {任务编号: $task_id})-[:具有环境]->(e:环境条件)
                RETURN e
                """, task_id=task_id)
            record = result.single()
            if record:
                return record['e']
            return None
            
    def get_constraint_data(self, task_id: str) -> Dict:
        """获取约束条件数据"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:任务 {任务编号: $task_id})-[:受约束]->(c:通信约束)
                RETURN c
                """, task_id=task_id)
            record = result.single()
            if record:
                return record['c']
            return None
            
    def get_similar_cases(self, task_id: str, limit: int = 10) -> List[Dict]:
        """获取相似历史案例"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t1:任务 {任务编号: $task_id})
                MATCH (t2:任务)
                WHERE t2.任务区域 = t1.任务区域 AND t2.任务编号 <> $task_id
                WITH t2, t1
                MATCH (t2)-[:具有环境]->(e2:环境条件)
                MATCH (t1)-[:具有环境]->(e1:环境条件)
                WITH t2, abs(e2.海况等级 - e1.海况等级) as sea_diff,
                     abs(toFloat(e2.电磁干扰强度) - toFloat(e1.电磁干扰强度)) as emi_diff
                ORDER BY sea_diff + emi_diff
                LIMIT $limit
                RETURN t2.任务编号 as task_id
                """, task_id=task_id, limit=limit)
            return [record['task_id'] for record in result]
            
    def save_optimization_results(self, task_id: str,
                                pareto_front: np.ndarray,
                                optimal_variables: np.ndarray) -> None:
        """保存优化结果"""
        with self.driver.session() as session:
            for i, (objectives, variables) in enumerate(zip(pareto_front, optimal_variables)):
                session.run("""
                    MATCH (t:任务 {任务编号: $task_id})
                    CREATE (r:优化结果 {
                        结果编号: $result_id,
                        可靠性目标: $reliability,
                        频谱效率目标: $spectral_efficiency,
                        能量效率目标: $energy_efficiency,
                        抗干扰目标: $interference,
                        环境适应性目标: $adaptability,
                        参数配置: $variables
                    })
                    CREATE (t)-[:优化方案]->(r)
                    """,
                    task_id=task_id,
                    result_id=f"{task_id}_result_{i}",
                    reliability=float(objectives[0]),
                    spectral_efficiency=float(objectives[1]),
                    energy_efficiency=float(objectives[2]),
                    interference=float(objectives[3]),
                    adaptability=float(objectives[4]),
                    variables=variables.tolist()
                )