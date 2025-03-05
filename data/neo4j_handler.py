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
                MATCH (t)-[:部署单位]->(n:节点)
                WITH t, collect(n) as nodes
                OPTIONAL MATCH (s:节点)-[r:通信手段]->(d:节点)
                WHERE s IN nodes AND d IN nodes
                RETURN t, nodes, collect(r) as relationships
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
    def process_task_record(self, record):
        """处理任务记录数据"""
        if not record:
            return None
            
        task = dict(record['t'])
        nodes = [dict(node) for node in record['nodes']]
        relationships = [dict(rel) for rel in record['relationships']]
        
        # 构造返回数据结构
        task_data = {
            'task_info': {
                'task_id': task.get('任务编号'),
                'task_name': task.get('任务名称'),
                'task_target': task.get('任务目标'),
                'task_area': task.get('任务区域'),
                'task_time': task.get('任务时间范围'),
                'force_composition': task.get('兵力组成'),
                'communication_plan': task.get('通信方案编号')
            },
            'nodes': {
                'command_center': None,    # 指挥所
                'command_ship': None,      # 海上指挥舰船
                'combat_units': [],        # 作战单位
                'comm_stations': [],       # 通信站
                'communication_systems': [] # 通信系统/设备
            },
            'communication_links': [],     # 通信链路
            'environment': None,           # 环境条件
            'constraints': None            # 通信约束
        }

        # 处理环境条件和约束条件
        for node in nodes:
            if '环境条件' in node.get('labels', []):
                task_data['environment'] = node
            elif '通信约束' in node.get('labels', []):
                task_data['constraints'] = node

        # 处理节点分类
        for node in nodes:
            node_type = node.get('properties', {}).get('节点类型')
            
            if node_type in ['航母', '驱逐舰', '护卫舰', '潜艇']:
                if node_type == '航母' or (node_type == '驱逐舰' and not task_data['nodes']['command_ship']):
                    task_data['nodes']['command_ship'] = node
                task_data['nodes']['combat_units'].append(node)
            elif '指挥所' in node.get('labels', []):
                task_data['nodes']['command_center'] = node
            elif '通信站' in node.get('labels', []):
                task_data['nodes']['comm_stations'].append(node)
            elif '通信设备' in node.get('labels', []):
                task_data['nodes']['communication_systems'].append(node)

        # 处理通信关系
        for rel in relationships:
            rel_type = rel.get('type')
            if rel_type == '通信手段':
                # 常规通信手段
                link = {
                    'source_id': rel['start'],
                    'target_id': rel['end'],
                    'comm_type': rel['properties'].get('通信手段类型'),
                    'frequency_band': rel['properties'].get('工作频段'),
                    'bandwidth': rel['properties'].get('带宽大小'),
                    'power': rel['properties'].get('发射功率'),
                    'required_equipment': rel['properties'].get('所需设备'),
                    'network_status': rel['properties'].get('网络状态'),
                    'path': rel['properties'].get('业务传输路径')
                }
                task_data['communication_links'].append(link)
            elif rel_type == '有线连接':
                # 有线连接（指挥所-通信站）
                link = {
                    'source_id': rel['start'],
                    'target_id': rel['end'],
                    'conn_type': '有线连接',
                    'line_type': rel['properties'].get('连接类型'),
                    'transmission_rate': rel['properties'].get('传输速率'),
                    'delay': rel['properties'].get('传输延迟')
                }
                task_data['communication_links'].append(link)

        return task_data