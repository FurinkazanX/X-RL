"""SAC 模型基类"""

from typing import List
from xrl.interfaces.model import Model
from xrl.core.types import Trajectory, Experience
import numpy as np
import torch
import torch.nn.functional as F


class SACModel(Model):
    """SAC 模型基类"""
    
    def __init__(self, **kwargs):
        """初始化 SAC 模型
        
        Args:
            **kwargs: 模型参数
        """
        super().__init__(**kwargs)
        self.gamma = kwargs.get('gamma', 0.99)
        self.alpha = kwargs.get('alpha', 0.2)
        self.target_entropy = kwargs.get('target_entropy', None)
        self.q1_optimizer = None
        self.q2_optimizer = None
        self.policy_optimizer = None
        self.alpha_optimizer = None
    
    def learn(self, batch) -> None:
        """更新模型参数
        
        Args:
            batch: SAC 批量数据
        """
        # 1. 提取批量数据
        states = batch.states
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_states
        dones = batch.dones
        
        # 2. 计算目标 Q 值
        next_model_output = self.forward({"state": next_states})
        next_actions = next_model_output.get("action", None)
        next_log_probs = next_model_output.get("log_prob", None)
        
        # 3. 计算目标 Q 值
        if next_actions is not None and next_log_probs is not None:
            next_q1 = self.q1_network(next_states, next_actions)
            next_q2 = self.q2_network(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_log_probs)
        else:
            target_q = rewards
        
        # 4. 计算 Q 网络损失
        current_q1 = self.q1_network(states, actions)
        current_q2 = self.q2_network(states, actions)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        # 5. 优化 Q 网络
        if self.q1_optimizer is not None and self.q2_optimizer is not None:
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()
            
            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()
        
        # 6. 计算策略网络损失
        policy_output = self.forward({"state": states})
        policy_actions = policy_output.get("action", None)
        policy_log_probs = policy_output.get("log_prob", None)
        
        if policy_actions is not None and policy_log_probs is not None:
            q1 = self.q1_network(states, policy_actions)
            q2 = self.q2_network(states, policy_actions)
            min_q = torch.min(q1, q2)
            policy_loss = (self.alpha * policy_log_probs - min_q).mean()
        else:
            policy_loss = 0.0
        
        # 7. 优化策略网络
        if self.policy_optimizer is not None:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
        # 8. 调整温度参数 alpha
        if self.alpha_optimizer is not None and policy_log_probs is not None:
            alpha_loss = -(self.alpha * (policy_log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
    
    @classmethod
    def process_trajectory(cls, trajectory: Trajectory, **kwargs) -> List[Experience]:
        """处理完整轨迹，返回处理后的经验数据列表
        
        Args:
            trajectory: 完整的轨迹数据
            **kwargs: 其他参数
        
        Returns:
            处理后的经验数据列表
        """
        # SAC 算法不需要对轨迹进行特殊处理，直接返回原始经验数据
        return trajectory.experiences

