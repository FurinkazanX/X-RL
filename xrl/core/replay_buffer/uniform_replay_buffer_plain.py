"""均匀采样 Replay Buffer 实现（普通版本，不被 @ray.remote 装饰）"""

import numpy as np
from typing import List, Any
from xrl.core.replay_buffer.base_replay_buffer import BaseReplayBuffer
from xrl.core.types import Experience, Batch


class UniformReplayBufferPlain(BaseReplayBuffer):
    """均匀采样 Replay Buffer 实现（普通版本）"""
    
    def __init__(self, size: int):
        super().__init__(size)
        self.buffer = []
        self.position = 0
    
    def add(self, experience: Experience) -> None:
        """添加经验数据
        
        Args:
            experience: 经验数据
        """
        if len(self.buffer) < self.size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.size
    
    def sample(self, batch_size: int) -> List[Experience]:
        """采样批量数据
        
        Args:
            batch_size: 批量大小
        
        Returns:
            经验数据列表
        """
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        experiences = [self.buffer[i] for i in indices]
        return experiences
    
    def get_size(self) -> int:
        """返回缓冲区当前大小"""
        return len(self.buffer)
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer = []
        self.position = 0
