"""Categorical 分布 - 离散动作"""

import torch
import torch.nn.functional as F
from xrl.algo.ppo.distribution.base import Distribution


class Categorical(Distribution):
    """Categorical 分布，用于离散动作空间"""
    
    def __init__(self, logits: torch.Tensor):
        """初始化
        
        Args:
            logits: 未归一化的对数概率 (batch_size, action_dim)
        """
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
    
    def sample(self) -> torch.Tensor:
        """采样动作"""
        return torch.multinomial(self.probs, 1).squeeze(-1)
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """计算对数概率"""
        log_probs = F.log_softmax(self.logits, dim=-1)
        # 确保 actions 是 long 类型
        actions = actions.long()
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    def entropy(self) -> torch.Tensor:
        """计算熵"""
        log_probs = F.log_softmax(self.logits, dim=-1)
        return -(self.probs * log_probs).sum(dim=-1)
    
    def kl(self, other: 'Categorical') -> torch.Tensor:
        """计算 KL 散度"""
        log_probs = F.log_softmax(self.logits, dim=-1)
        other_log_probs = F.log_softmax(other.logits, dim=-1)
        return (self.probs * (log_probs - other_log_probs)).sum(dim=-1)
    
    def mode(self) -> torch.Tensor:
        """返回众数（贪心选择）"""
        return torch.argmax(self.probs, dim=-1)
