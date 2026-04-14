"""Env 接口定义"""

from typing import Dict, Any, Tuple


class Env:
    """环境接口"""
    
    def reset(self) -> Dict[str, Any]:
        """重置环境，返回初始状态
        
        Returns:
            初始状态
        """
        raise NotImplementedError
    
    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """执行动作，返回下一个状态、奖励、是否结束、额外信息
        
        Args:
            actions: 智能体动作
        
        Returns:
            下一个状态、奖励、是否结束、额外信息
        """
        raise NotImplementedError
    
    def close(self) -> None:
        """关闭环境"""
        raise NotImplementedError
