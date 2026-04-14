"""Controller 基类接口"""

from typing import Dict, Any
import time
import psutil
import ray


class BaseController:
    """Controller 基类"""
    
    def __init__(self, config):
        """初始化 Controller
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.components = {}
        self.running = False
        
        # 获取 controller 配置
        controller_config = config.get("controller", {})
        
        # 调度参数
        self.min_buffer_size = controller_config.get("min_buffer_size", 1000)
        self.max_buffer_size = controller_config.get("max_buffer_size", 100000)
        self.target_buffer_size = controller_config.get("target_buffer_size", 10000)
        self.train_interval = controller_config.get("train_interval", 1.0)  # 训练间隔（秒）
        self.episode_interval = controller_config.get("episode_interval", 0.1)  # episode 间隔（秒）
        
        # 资源监控参数
        self.cpu_threshold = controller_config.get("cpu_threshold", 0.8)
        self.memory_threshold = controller_config.get("memory_threshold", 0.8)
        
        # 状态跟踪
        self.last_train_time = time.time()
        self.last_episode_time = time.time()
        self.episode_count = 0
        self.train_count = 0
    
    def should_train(self) -> bool:
        """判断是否应该训练
        
        Returns:
            是否应该训练
        """
        # 检查训练间隔
        current_time = time.time()
        if current_time - self.last_train_time < self.train_interval:
            return False
        
        # 检查系统资源使用情况
        cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
        memory_usage = psutil.virtual_memory().percent / 100.0
        
        # 如果资源使用率过高，延迟训练
        if cpu_usage > self.cpu_threshold or memory_usage > self.memory_threshold:
            return False
        
        return True
    
    def should_run_episode(self) -> bool:
        """判断是否应该运行 episode
        
        Returns:
            是否应该运行 episode
        """
        # 检查 episode 间隔
        current_time = time.time()
        if current_time - self.last_episode_time < self.episode_interval:
            return False
        
        # 检查缓冲区大小
        if "replay_buffer" in self.components:
            buffer_size = self.get_buffer_size()
            if buffer_size >= self.max_buffer_size:
                # 缓冲区已满，暂停数据收集
                return False
        
        return True
    
    def get_buffer_size(self) -> int:
        """获取缓冲区大小
        
        Returns:
            缓冲区大小
        """
        if "replay_buffer" in self.components:
            try:
                return ray.get(self.components["replay_buffer"].get_size.remote())
            except Exception as e:
                return 0
        return 0
    
    def initialize(self):
        """初始化所有组件"""
        pass
    
    def start(self):
        """启动所有组件"""
        pass
    
    def stop(self):
        """停止所有组件"""
        self.running = False
    
    def monitor(self):
        """监控系统运行状态"""
        pass
