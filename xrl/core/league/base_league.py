"""League 基类接口"""

from typing import Dict, Any, Optional


class BaseLeague:
    """League 基类"""
    
    def __init__(self, config):
        """初始化联赛
        
        Args:
            config: 联赛配置
        """
        self.config = config
        self.agent_pool = {}
    
    def add_agent(self, agent_id, agent):
        """添加智能体到联赛
        
        Args:
            agent_id: 智能体唯一标识符
            agent: 智能体实例
        """
        pass
    
    def remove_agent(self, agent_id):
        """从联赛中移除智能体
        
        Args:
            agent_id: 智能体唯一标识符
        """
        pass
    
    def run_season(self):
        """运行一个赛季的比赛"""
        pass
    
    def evaluate_agents(self):
        """评估所有智能体的性能"""
        pass
    
    def select_agents(self):
        """选择表现优秀的智能体"""
        pass
    
    def update_agents(self):
        """更新智能体池，替换表现差的智能体"""
        pass
