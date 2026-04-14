"""优先级采样 Replay Buffer 实现"""

import ray
import numpy as np
from typing import List, Any
from xrl.core.replay_buffer.base_replay_buffer import BaseReplayBuffer
from xrl.core.types import Experience, Batch


@ray.remote
class PrioritizedReplayBuffer(BaseReplayBuffer):
    """优先级采样 Replay Buffer 实现"""
    
    def __init__(self, size: int, alpha: float = 0.6, beta: float = 0.4):
        super().__init__(size)
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.alpha = alpha  # 优先级指数
        self.beta = beta  # 重要性采样权重指数
        self.beta_increment = 0.001  # beta 增量
        self.max_priority = 1.0  # 最大优先级
    
    def add(self, experience: Experience) -> None:
        """添加经验数据
        
        Args:
            experience: 经验数据
        """
        if len(self.buffer) < self.size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # 设置优先级
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.size
    
    def sample(self, batch_size: int) -> Batch:
        """采样批量数据
        
        Args:
            batch_size: 批量大小
        
        Returns:
            批量经验数据
        """
        buffer_size = len(self.buffer)
        if buffer_size == 0:
            raise ValueError("Replay buffer is empty")
        
        # 计算优先级概率
        priorities = self.priorities[:buffer_size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 采样
        indices = np.random.choice(buffer_size, size=batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]
        
        # 计算重要性采样权重
        weights = (buffer_size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # 构建批量数据
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])
        infos = {
            "indices": indices,
            "weights": weights
        }
        
        # 收集额外信息
        for i, exp in enumerate(experiences):
            for key, value in exp.info.items():
                if key not in infos:
                    infos[key] = []
                infos[key].append(value)
        
        # 更新 beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return Batch(states, actions, rewards, next_states, dones, infos)
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """更新优先级
        
        Args:
            indices: 经验数据索引
            priorities: 新的优先级
        """
        for i, idx in enumerate(indices):
            self.priorities[idx] = priorities[i]
            if priorities[i] > self.max_priority:
                self.max_priority = priorities[i]
    
    def get_size(self) -> int:
        """返回缓冲区当前大小"""
        return len(self.buffer)
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((self.size,), dtype=np.float32)
        self.max_priority = 1.0
