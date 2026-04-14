"""DQN 模型基类"""

from typing import List
from xrl.interfaces.model import Model
from xrl.types import DQNExperience
from xrl.core.types import Trajectory, Experience
import numpy as np
import torch
import torch.nn.functional as F


class DQNModel(Model):
    """DQN 模型基类"""
    
    def __init__(self, **kwargs):
        """初始化 DQN 模型
        
        Args:
            **kwargs: 模型参数
        """
        super().__init__(**kwargs)
        self.gamma = kwargs.get('gamma', 0.99)
        self.target_model = None
        self.optimizer = None
    
    def learn(self, batch) -> None:
        """更新模型参数
        
        Args:
            batch: DQN 批量数据
        """
        # 1. 提取批量数据
        states = batch.states
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_states
        dones = batch.dones
        
        # 2. 计算当前 Q 值
        model_output = self.forward({"state": states})
        current_q = model_output.get("q_values", None)
        
        # 3. 计算目标 Q 值
        if self.target_model is not None:
            target_output = self.target_model.forward({"state": next_states})
            next_q = target_output.get("q_values", None)
            if next_q is not None:
                max_next_q = torch.max(next_q, dim=1, keepdim=True)[0]
                target_q = rewards + self.gamma * max_next_q * (1 - dones)
            else:
                target_q = rewards
        else:
            # 如果没有目标模型，使用当前模型计算目标 Q 值
            target_output = self.forward({"state": next_states})
            next_q = target_output.get("q_values", None)
            if next_q is not None:
                max_next_q = torch.max(next_q, dim=1, keepdim=True)[0]
                target_q = rewards + self.gamma * max_next_q * (1 - dones)
            else:
                target_q = rewards
        
        # 4. 计算损失
        if current_q is not None and target_q is not None:
            # 只考虑实际执行的动作的 Q 值
            actions = actions.long().unsqueeze(1)
            current_q = torch.gather(current_q, 1, actions)
            loss = F.mse_loss(current_q, target_q)
        else:
            loss = 0.0
        
        # 5. 优化模型参数
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    @classmethod
    def process_trajectory(cls, trajectory: Trajectory, **kwargs) -> List[Experience]:
        """处理完整轨迹，返回处理后的经验数据列表
        
        Args:
            trajectory: 完整的轨迹数据
            **kwargs: 其他参数
        
        Returns:
            处理后的经验数据列表
        """
        # DQN 算法不需要对轨迹进行特殊处理，直接返回原始经验数据
        return trajectory.experiences

