"""智能体选择器实现"""

from typing import Dict, Any, List


class BaseSelector:
    """选择器基类"""
    
    def __init__(self, config):
        """初始化选择器
        
        Args:
            config: 选择器配置
        """
        self.config = config
    
    def select(self, agent_pool: Dict[str, Any]) -> List[str]:
        """选择表现优秀的智能体
        
        Args:
            agent_pool: 智能体池
        
        Returns:
            选中的智能体 ID 列表
        """
        # 实现选择逻辑，返回选中的智能体
        sorted_agents = sorted(
            agent_pool.items(),
            key=lambda x: x[1]['performance'],
            reverse=True
        )
        
        top_k = self.config.get('top_k', 5)
        return [agent_id for agent_id, _ in sorted_agents[:top_k]]
