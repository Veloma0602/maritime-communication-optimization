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
            # 直接采用更简单的查询方式，与测试查询保持一致
            with self.driver.session() as session:
                # 获取任务
                task_result = session.run("""
                    MATCH (t:任务 {任务编号: $task_id})
                    RETURN t
                    """, task_id=task_id)
                
                task_record = task_result.single()
                if not task_record:
                    logger.warning(f"No task found with ID: {task_id}")
                    return None
                
                task = task_record['t']
                
                # 获取任务部署的节点
                node_result = session.run("""
                    MATCH (t:任务 {任务编号: $task_id})
                    MATCH (t)-[:部署单位]->(n:节点)
                    RETURN collect(n) as nodes
                    """, task_id=task_id)
                
                node_record = node_result.single()
                nodes = node_record['nodes'] if node_record else []
                
                relation_result = session.run("""
                    MATCH (t:任务 {任务编号: $task_id})
                    MATCH (t)-[:部署单位]->(n1:节点)
                    MATCH (n1)-[r:通信手段]->(n2:节点)
                    WHERE (t)-[:部署单位]->(n2)
                    RETURN r, type(r) as rel_type, id(n1) as start_id, id(n2) as end_id
                    """, task_id=task_id)

                relationships = []
                for record in relation_result:
                    rel = dict(record['r'])
                    # 添加类型信息到关系字典
                    rel['type'] = record['rel_type']
                    rel['start'] = record['start_id']
                    rel['end'] = record['end_id']
                    relationships.append(rel)
                    
                print(f"Retrieved {len(relationships)} communication relations with type info")
                
                # 构建记录
                record = {
                    't': task,
                    'nodes': nodes,
                    'relationships': relationships
                }
                
                # 处理记录
                task_data = self.process_task_record(record)
                
                logger.info(f"Retrieved task data for {task_id}: {len(task_data['communication_links'])} links")
                
                if not task_data['communication_links']:
                    logger.warning(f"No communication links found for task {task_id}")
                    logger.warning(f"Task has {len(nodes)} deployed nodes")
                
                return task_data
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
        """获取相似历史案例"""
        try:
            with self.driver.session() as session:
                # 基于任务区域和兵力组成寻找相似任务
                result = session.run("""
                    MATCH (t1:任务 {任务编号: $task_id})
                    MATCH (t2:任务)
                    WHERE t2.任务编号 <> $task_id
                    
                    WITH t1, t2,
                        CASE 
                            WHEN t1.任务区域 = t2.任务区域 THEN 0
                            ELSE 1
                        END as area_diff,
                        CASE
                            WHEN t1.兵力组成 = t2.兵力组成 THEN 0
                            ELSE 1
                        END as force_diff
                    
                    WITH t2, (area_diff + force_diff) as total_diff
                    ORDER BY total_diff ASC
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
        
        # 调试信息
        print(f"处理任务记录: 找到 {len(nodes)} 个节点, {len(relationships)} 个关系")
        
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
            node_type = node.get('节点类型', '')
            
            if node_type in ['航母', '驱逐舰', '护卫舰', '潜艇']:
                if node_type == '航母' or (node_type == '驱逐舰' and 
                                        not task_data['nodes']['command_ship']):
                    task_data['nodes']['command_ship'] = node
                task_data['nodes']['combat_units'].append(node)
            elif '指挥所' in node.get('labels', []) or node_type == '指挥所':
                task_data['nodes']['command_center'] = node
            elif '通信站' in node.get('labels', []) or node_type == '通信站':
                task_data['nodes']['comm_stations'].append(node)
            elif '通信设备' in node.get('labels', []) or node_type == '通信设备':
                task_data['nodes']['communication_systems'].append(node)
        
        # 关系处理部分
        for rel in relationships:
            # 调试输出关系信息
            rel_type = rel.get('type', '未知')
            print(f"处理关系: type={rel_type}, properties={rel.get('properties', {})}")
            
            # 检查是否是通信手段关系 - 更宽松的匹配
            if '通信' in rel_type or rel_type == '通信手段':
                # 获取节点之间的关系属性
                properties = rel.get('properties', {})
                
                # 构建通信链路信息
                link = {
                    'source_id': rel.get('start'),
                    'target_id': rel.get('end'),
                    'comm_type': properties.get('通信手段类型', '未知'),
                    'frequency_band': properties.get('工作频段', ''),
                    'bandwidth': self._parse_bandwidth(properties.get('带宽大小', '')),
                    'power': self._parse_power(properties.get('发射功率', '')),
                    'required_equipment': properties.get('所需设备', ''),
                    'network_status': properties.get('网络状态', ''),
                    'path': properties.get('业务传输路径', '')
                }
                
                # 解析频率
                freq_min, freq_max = self._parse_frequency_band(properties.get('工作频段', ''))
                link['frequency_min'] = freq_min
                link['frequency_max'] = freq_max
                link['frequency'] = (freq_min + freq_max) / 2  # 使用中心频率
                
                task_data['communication_links'].append(link)
                print(f"  创建通信链路: {link['source_id']} -> {link['target_id']}, 类型={link['comm_type']}")
                
            elif rel_type == '有线连接':
                # 处理有线连接
                properties = rel.get('properties', {})
                link = {
                    'source_id': rel.get('start'),
                    'target_id': rel.get('end'),
                    'conn_type': '有线连接',
                    'line_type': properties.get('连接类型', ''),
                    'transmission_rate': properties.get('传输速率', ''),
                    'delay': properties.get('传输延迟', '')
                }
                task_data['communication_links'].append(link)

        print(f"处理完成: 创建了 {len(task_data['communication_links'])} 个通信链路")
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
            min_freq = self._parse_numeric_value(parts[0].strip())
            max_freq = self._parse_numeric_value(parts[1].strip())
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
        return self._parse_numeric_value(bandwidth)
        
    # def _parse_power(self, power: str) -> float:
    #     """
    #     解析功率字符串，提取数值
        
    #     参数:
    #     power: 功率字符串（如 "33w"、"33W"）
        
    #     返回:
    #     功率值（W）
    #     """
    #     try:
    #         if not power:
    #             return 0
                
    #         # 移除单位并转换为浮点数
    #         value = ''.join(c for c in power if c.isdigit() or c == '.')
    #         return float(value) if value else 0
    #     except Exception:
    #         return 0
    def _parse_power(self, power: str) -> float:
        """
        解析功率字符串，提取数值
        
        参数:
        power: 功率字符串（如 "33w"、"33W"）
        
        返回:
        功率值（W）
        """
        return self._parse_numeric_value(power)

    def get_historical_communication_parameters(self, task_id: str) -> List[Dict]:
        """从历史任务中获取通信参数配置"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (t:任务 {任务编号: $task_id})
                    MATCH (t)-[:部署单位]->(n1:节点)
                    MATCH (n1)-[r:通信手段]->(n2:节点)
                    WHERE (t)-[:部署单位]->(n2)
                    RETURN r as communication_link
                    """, task_id=task_id)
                
                parameters = []
                for record in result:
                    link = dict(record['communication_link'])
                    link_params = self._extract_communication_parameters(link)
                    if link_params:
                        parameters.append(link_params)
                
                return parameters
        except Exception as e:
            logger.error(f"Error retrieving communication parameters for {task_id}: {str(e)}")
            return []

    
    def _extract_communication_parameters(self, link: Dict) -> Optional[Dict]:
        """从通信链路中提取通信参数"""
        try:
            # 提取频率参数
            freq_band = link.get('properties', {}).get('工作频段', '')
            freq_min, freq_max = self._parse_frequency_band(freq_band)
            center_freq = (freq_min + freq_max) / 2 if freq_min and freq_max else 0
            
            # 提取带宽参数
            bandwidth_text = link.get('properties', {}).get('带宽大小', '')
            bandwidth = self._parse_bandwidth(bandwidth_text)
            
            # 提取功率参数
            power_text = link.get('properties', {}).get('发射功率', '')
            power = self._parse_power(power_text)
            
            # 提取调制方式(可能需要从设备或默认值推断)
            comm_type = link.get('properties', {}).get('通信手段类型', '')
            modulation = self._infer_modulation(comm_type)
            
            # 提取极化方式(可能需要从设备或默认值推断)
            polarization = self._infer_polarization(comm_type)
            
            return {
                'frequency': center_freq,
                'bandwidth': bandwidth,
                'power': power,
                'modulation': modulation,
                'polarization': polarization,
                'source_id': link.get('start'),
                'target_id': link.get('end'),
                'link_type': comm_type
            }
        except Exception as e:
            logger.error(f"Error extracting parameters: {str(e)}")
            return None

    def _infer_modulation(self, comm_type: str) -> str:
        """根据通信方式推断调制方式"""
        modulation_map = {
            '卫星通信': 'QPSK',
            '超低频通信': 'BPSK',
            '短波通信': 'BPSK',
            '数据链': 'QAM16'
        }
        for key, value in modulation_map.items():
            if key in comm_type:
                return value
        return 'BPSK'  # 默认值

    def _infer_polarization(self, comm_type: str) -> str:
        """根据通信方式推断极化方式"""
        if '卫星' in comm_type:
            return 'CIRCULAR'
        elif '低频' in comm_type:
            return 'LINEAR'
        else:
            return 'LINEAR'  # 默认值

    def get_task_communication_links(self, task_id: str) -> List[Dict]:
        """获取特定任务的通信链路，确保链路确实属于该任务"""
        try:
            with self.driver.session() as session:
                # 先获取任务部署的所有节点
                node_result = session.run("""
                    MATCH (t:任务 {任务编号: $task_id})
                    MATCH (t)-[:部署单位]->(n:节点)
                    RETURN collect(n.编号) as task_nodes
                    """, task_id=task_id)
                
                task_node_record = node_result.single()
                if not task_node_record:
                    return []
                    
                task_nodes = set(task_node_record['task_nodes'])
                
                # 然后获取这些节点之间的通信链路
                link_result = session.run("""
                    MATCH (s:节点)-[r:通信手段]->(t:节点)
                    WHERE s.编号 IN $task_nodes AND t.编号 IN $task_nodes
                    RETURN r
                    """, task_nodes=list(task_nodes))
                
                links = []
                for record in link_result:
                    link_data = dict(record['r'])
                    # 验证链路确实适用于当前任务
                    if self._verify_link_for_task(link_data, task_id):
                        links.append(link_data)

                logger.info(f"Retrieved {len(links)} communication links for task {task_id}")
                return links
        except Exception as e:
            logger.error(f"Error retrieving communication links: {str(e)}")
            return []
            
    def _verify_link_for_task(self, link: Dict, task_id: str) -> bool:
        """验证通信链路是否适用于特定任务"""
        # 简单实现：假设所有在任务节点间的链路都属于该任务
        return True

    def _parse_numeric_value(self, value_str: str) -> float:
        """
        从可能包含单位的字符串中提取数值
        
        参数:
        value_str: 可能包含单位的数值字符串
        
        返回:
        提取的浮点数值
        """
        if value_str is None:
            return 0.0
            
        if isinstance(value_str, (int, float)):
            return float(value_str)
        
        if isinstance(value_str, str):
            # 尝试直接转换
            try:
                return float(value_str)
            except ValueError:
                pass
                
            # 提取数字部分 (包括负号和小数点)
            import re
            numeric_match = re.search(r'-?\d+\.?\d*', value_str)
            if numeric_match:
                try:
                    return float(numeric_match.group())
                except ValueError:
                    pass
        
        # 默认返回0
        return 0.0

    # 测试方法
    def test_query(self, task_id: str):
        """测试查询，用于调试"""
        try:
            with self.driver.session() as session:
                # 测试1: 简单查询任务
                task_result = session.run("""
                    MATCH (t:任务 {任务编号: $task_id})
                    RETURN t
                    """, task_id=task_id)
                
                task_record = task_result.single()
                print(f"Test 1 - Task query: {'Success' if task_record else 'Failed'}")
                
                # 测试2: 查询任务部署的节点
                node_result = session.run("""
                    MATCH (t:任务 {任务编号: $task_id})
                    MATCH (t)-[:部署单位]->(n:节点)
                    RETURN count(n) as node_count
                    """, task_id=task_id)
                
                node_record = node_result.single()
                node_count = node_record['node_count'] if node_record else 0
                print(f"Test 2 - Task nodes: {node_count}")

                # 详细查看关系结构
                relation_detail_result = session.run("""
                    MATCH (t:任务 {任务编号: $task_id})
                    MATCH (t)-[:部署单位]->(n1:节点)
                    MATCH (n1)-[r:通信手段]->(n2:节点)
                    WHERE (t)-[:部署单位]->(n2)
                    RETURN n1.编号 as source, n2.编号 as target, type(r) as rel_type, r as rel_data
                    LIMIT 1
                    """, task_id=task_id)

                detail_record = relation_detail_result.single()
                if detail_record:
                    print("\nDetailed relation structure:")
                    print(f"Source: {detail_record['source']}")
                    print(f"Target: {detail_record['target']}")
                    print(f"Relation type: {detail_record['rel_type']}")
                    print(f"Relation data: {dict(detail_record['rel_data'])}")
                    
                    # 检查关系的具体属性
                    rel_data = dict(detail_record['rel_data'])
                    print("\nRelation properties structure:")
                    for key, value in rel_data.items():
                        print(f"  {key}: {value} (type: {type(value).__name__})")
                
                # 测试3: 查询节点间的通信关系
                if node_count > 0:
                    relation_result = session.run("""
                        MATCH (t:任务 {任务编号: $task_id})
                        MATCH (t)-[:部署单位]->(n1:节点)
                        MATCH (n1)-[r:通信手段]->(n2:节点)
                        WHERE (t)-[:部署单位]->(n2)
                        RETURN count(r) as rel_count
                        """, task_id=task_id)
                    
                    relation_record = relation_result.single()
                    rel_count = relation_record['rel_count'] if relation_record else 0
                    print(f"Test 3 - Communication relations: {rel_count}")
                    
                    # 测试4: 如果没有找到通信关系，尝试更宽松的查询
                    if rel_count == 0:
                        all_rel_result = session.run("""
                            MATCH (n1:节点)-[r:通信手段]->(n2:节点)
                            RETURN count(r) as all_rel_count
                            """)
                        
                        all_rel_record = all_rel_result.single()
                        all_rel_count = all_rel_record['all_rel_count'] if all_rel_record else 0
                        print(f"Test 4 - All communication relations in DB: {all_rel_count}")
                        
                        # 输出一些示例通信关系
                        if all_rel_count > 0:
                            sample_result = session.run("""
                                MATCH (n1:节点)-[r:通信手段]->(n2:节点)
                                RETURN n1.编号 as source, n2.编号 as target, r
                                LIMIT 3
                                """)
                            
                            print("Sample communication relations:")
                            for record in sample_result:
                                print(f"  {record['source']} -> {record['target']}")
                            
                return True
        except Exception as e:
            print(f"Test query error: {str(e)}")
            return False