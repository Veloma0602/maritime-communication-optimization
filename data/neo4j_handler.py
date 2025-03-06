from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional, Union
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jHandler:
    def __init__(self, uri: str, user: str, password: str):
        """
        初始化Neo4j处理器
        
        参数:
        uri: Neo4j数据库URI
        user: 用户名
        password: 密码
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info(f"Successfully connected to Neo4j at {uri}")
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value != 1:
                    raise Exception("Connection test failed")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
        
    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'driver'):
            self.driver.close()
            logger.info("Neo4j connection closed")
        
    def get_task_data(self, task_id: str) -> Optional[Dict]:
        """
        获取任务数据
        
        参数:
        task_id: 任务ID
        
        返回:
        任务数据字典或None（如果找不到）
        """
        try:
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
                    task_data = self.process_task_record(record)
                    logger.info(f"Retrieved task data for {task_id}: {len(task_data['communication_links'])} links")
                    return task_data
                
                logger.warning(f"No task data found for task ID: {task_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving task data for {task_id}: {str(e)}")
            return None
                
    def get_environment_data(self, task_id: str) -> Optional[Dict]:
        """
        获取环境条件数据
        
        参数:
        task_id: 任务ID
        
        返回:
        环境数据字典或None（如果找不到）
        """
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (t:任务 {任务编号: $task_id})-[:具有环境]->(e:环境条件)
                    RETURN e
                    """, task_id=task_id)
                
                record = result.single()
                if record:
                    env_data = dict(record['e'])
                    logger.info(f"Retrieved environment data for {task_id}")
                    return env_data
                
                logger.warning(f"No environment data found for task ID: {task_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving environment data for {task_id}: {str(e)}")
            return None
            
    def get_constraint_data(self, task_id: str) -> Optional[Dict]:
        """
        获取约束条件数据
        
        参数:
        task_id: 任务ID
        
        返回:
        约束数据字典或None（如果找不到）
        """
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (t:任务 {任务编号: $task_id})-[:受约束]->(c:通信约束)
                    RETURN c
                    """, task_id=task_id)
                
                record = result.single()
                if record:
                    constraint_data = dict(record['c'])
                    logger.info(f"Retrieved constraint data for {task_id}")
                    return constraint_data
                
                logger.warning(f"No constraint data found for task ID: {task_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving constraint data for {task_id}: {str(e)}")
            return None
            
    def get_similar_cases(self, task_id: str, limit: int = 10) -> List[str]:
        """
        获取相似历史案例
        
        参数:
        task_id: 当前任务ID
        limit: 最大返回案例数
        
        返回:
        相似任务ID列表
        """
        try:
            with self.driver.session() as session:
                # 基于环境条件和任务区域的相似性寻找历史案例
                result = session.run("""
                    MATCH (t1:任务 {任务编号: $task_id})
                    MATCH (t2:任务)
                    WHERE t2.任务区域 = t1.任务区域 AND t2.任务编号 <> $task_id
                    
                    // 获取环境条件相似度
                    WITH t1, t2
                    OPTIONAL MATCH (t1)-[:具有环境]->(e1:环境条件)
                    OPTIONAL MATCH (t2)-[:具有环境]->(e2:环境条件)
                    
                    WITH t2, 
                         CASE WHEN e1 IS NULL OR e2 IS NULL THEN 10
                              ELSE abs(toFloat(e1.海况等级) - toFloat(e2.海况等级)) +
                                   abs(toFloat(e1.电磁干扰强度) - toFloat(e2.电磁干扰强度))
                         END as env_diff
                    
                    // 按环境差异排序
                    ORDER BY env_diff
                    LIMIT $limit
                    
                    RETURN t2.任务编号 as task_id
                    """, task_id=task_id, limit=limit)
                
                similar_cases = [record['task_id'] for record in result]
                logger.info(f"Found {len(similar_cases)} similar cases for task {task_id}")
                return similar_cases
                
        except Exception as e:
            logger.error(f"Error finding similar cases for {task_id}: {str(e)}")
            return []
    
    def get_optimization_results(self, task_id: str, limit: int = 5) -> List[Dict]:
        """
        获取历史优化结果
        
        参数:
        task_id: 任务ID
        limit: 最大返回结果数
        
        返回:
        优化结果列表
        """
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (t:任务 {任务编号: $task_id})-[:优化方案]->(r:优化结果)
                    RETURN r
                    ORDER BY r.可靠性目标 DESC
                    LIMIT $limit
                    """, task_id=task_id, limit=limit)
                
                optimization_results = []
                for record in result:
                    result_node = dict(record['r'])
                    optimization_results.append(result_node)
                
                logger.info(f"Retrieved {len(optimization_results)} optimization results for task {task_id}")
                return optimization_results
                
        except Exception as e:
            logger.error(f"Error retrieving optimization results for {task_id}: {str(e)}")
            return []
            
    def save_optimization_results(self, task_id: str,
                                pareto_front: np.ndarray,
                                optimal_variables: np.ndarray) -> None:
        """
        保存优化结果到数据库
        
        参数:
        task_id: 任务ID
        pareto_front: Pareto前沿（目标函数值）
        optimal_variables: 对应的最优变量值
        """
        try:
            with self.driver.session() as session:
                # 先清除可能存在的旧结果
                session.run("""
                    MATCH (t:任务 {任务编号: $task_id})-[r:优化方案]->(result:优化结果)
                    DELETE r, result
                    """, task_id=task_id)
                
                # 保存新结果
                for i, (objectives, variables) in enumerate(zip(pareto_front, optimal_variables)):
                    # 每个解的唯一标识
                    result_id = f"{task_id}_result_{i}"
                    
                    # 解的目标函数值
                    reliability = float(-objectives[0])  # 注意取反，因为优化过程是最小化目标
                    spectral_efficiency = float(-objectives[1])
                    energy_efficiency = float(-objectives[2])
                    interference = float(-objectives[3])
                    adaptability = float(-objectives[4])
                    
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
                        result_id=result_id,
                        reliability=reliability,
                        spectral_efficiency=spectral_efficiency,
                        energy_efficiency=energy_efficiency,
                        interference=interference,
                        adaptability=adaptability,
                        variables=variables.tolist()
                    )
                
                logger.info(f"Saved {len(pareto_front)} optimization results for task {task_id}")
        
        except Exception as e:
            logger.error(f"Error saving optimization results for {task_id}: {str(e)}")
            raise
    
    def process_task_record(self, record):
        """
        处理任务记录数据，构建标准化的任务数据结构
        
        参数:
        record: 从Neo4j查询返回的记录
        
        返回:
        处理后的任务数据字典
        """
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

        # 处理节点分类
        for node in nodes:
            node_type = node.get('节点类型')
            
            if node_type in ['航母', '驱逐舰', '护卫舰', '潜艇']:
                if node_type == '航母' or (node_type == '驱逐舰' and 
                                        not task_data['nodes']['command_ship']):
                    task_data['nodes']['command_ship'] = node
                task_data['nodes']['combat_units'].append(node)
            elif node.get('labels', []) and '指挥所' in node.get('labels', []):
                task_data['nodes']['command_center'] = node
            elif node.get('labels', []) and '通信站' in node.get('labels', []):
                task_data['nodes']['comm_stations'].append(node)
            elif node.get('labels', []) and '通信设备' in node.get('labels', []):
                task_data['nodes']['communication_systems'].append(node)

        # 处理通信关系，标准化通信链路数据
        for rel in relationships:
            rel_type = rel.get('type')
            if rel_type == '通信手段':
                # 提取并标准化链路属性
                freq_band = rel.get('properties', {}).get('工作频段', '')
                bandwidth = rel.get('properties', {}).get('带宽大小', '')
                power = rel.get('properties', {}).get('发射功率', '')
                
                # 尝试将文本频段和带宽转换为数值
                freq_min, freq_max = self._parse_frequency_band(freq_band)
                bandwidth_value = self._parse_bandwidth(bandwidth)
                power_value = self._parse_power(power)
                
                # 构建通信链路数据
                link = {
                    'source_id': rel.get('start'),
                    'target_id': rel.get('end'),
                    'comm_type': rel.get('properties', {}).get('通信手段类型'),
                    'frequency_min': freq_min,
                    'frequency_max': freq_max,
                    'frequency_band': freq_band,
                    'bandwidth': bandwidth_value,
                    'bandwidth_text': bandwidth,
                    'power': power_value,
                    'power_text': power,
                    'required_equipment': rel.get('properties', {}).get('所需设备'),
                    'network_status': rel.get('properties', {}).get('网络状态'),
                    'path': rel.get('properties', {}).get('业务传输路径', '')
                }
                task_data['communication_links'].append(link)
            
            elif rel_type == '有线连接':
                # 处理有线连接
                link = {
                    'source_id': rel.get('start'),
                    'target_id': rel.get('end'),
                    'conn_type': '有线连接',
                    'line_type': rel.get('properties', {}).get('连接类型'),
                    'transmission_rate': rel.get('properties', {}).get('传输速率'),
                    'delay': rel.get('properties', {}).get('传输延迟')
                }
                task_data['communication_links'].append(link)

        return task_data
    
    def _parse_frequency_band(self, freq_band: str) -> tuple:
        """
        解析频段字符串，提取最小和最大频率值
        
        参数:
        freq_band: 频段字符串（如 "3000/4300"）
        
        返回:
        (min_freq, max_freq): 频率范围（MHz）
        """
        try:
            if not freq_band or '/' not in freq_band:
                return 0, 0
                
            parts = freq_band.split('/')
            min_freq = float(parts[0].strip())
            max_freq = float(parts[1].strip())
            return min_freq, max_freq
        except Exception:
            return 0, 0
    
    def _parse_bandwidth(self, bandwidth: str) -> float:
        """
        解析带宽字符串，提取数值
        
        参数:
        bandwidth: 带宽字符串（如 "1300"、"1300MHz"）
        
        返回:
        带宽值（MHz）
        """
        try:
            if not bandwidth:
                return 0
                
            # 移除单位并转换为浮点数
            value = ''.join(c for c in bandwidth if c.isdigit() or c == '.')
            return float(value) if value else 0
        except Exception:
            return 0
    
    def _parse_power(self, power: str) -> float:
        """
        解析功率字符串，提取数值
        
        参数:
        power: 功率字符串（如 "33w"、"33W"）
        
        返回:
        功率值（W）
        """
        try:
            if not power:
                return 0
                
            # 移除单位并转换为浮点数
            value = ''.join(c for c in power if c.isdigit() or c == '.')
            return float(value) if value else 0
        except Exception:
            return 0