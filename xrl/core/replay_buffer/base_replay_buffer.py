"""Replay Buffer 基类接口"""

from typing import List, Any, Optional
from xrl.core.types import Experience


class BaseReplayBuffer:
    """Replay Buffer 基类"""
    
    def __init__(self, size: int):
        """初始化 Replay Buffer
        
        Args:
            size: 缓冲区大小
        """
        self.size = size
    
    def add(self, experience: Experience) -> None:
        """添加经验数据
        
        Args:
            experience: 经验数据
        """
        raise NotImplementedError
    
    def sample(self, batch_size: int) -> List[Experience]:
        """采样批量数据
        
        Args:
            batch_size: 批量大小
        
        Returns:
            经验数据列表
        """
        raise NotImplementedError
    
    def get_size(self) -> int:
        """返回缓冲区当前大小"""
        raise NotImplementedError
    
    def clear(self) -> None:
        """清空缓冲区"""
        raise NotImplementedError
