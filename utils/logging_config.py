import logging
import os
from datetime import datetime

def setup_logging(log_dir="logs"):
    """
    设置基于时间戳的日志文件名的日志系统
    
    参数:
    log_dir: 存储日志文件的目录
    
    返回:
    日志记录器实例
    """
    # 如果日志目录不存在则创建
    os.makedirs(log_dir, exist_ok=True)
    
    # 为日志文件名创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"optimization_{timestamp}.log")
    
    # 确保之前的处理器被移除
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建并配置新的处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    # 设置格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到根记录器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    # 创建一个命名记录器用于当前模块
    logger = logging.getLogger('optimization')
    logger.info(f"日志系统初始化完成。日志文件: {log_filename}")
    
    return logger