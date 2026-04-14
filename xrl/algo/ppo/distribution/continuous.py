"""连续高斯分布 - 支持高维"""

import torch
import torch.distributions as D
from xrl.algo.ppo.distribution.base import Distribution


class Continuous(Distribution):
    """连续高斯分布，用于连续动作空间，支持高维"""
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """初始化
        
        Args:
            mean: 均值 (batch_size, action_dim)
            std: 标准差 (batch_size, action_dim) 或 (action_dim,)
        """
        self.mean = mean
        
        # 如果 std 是一维的，扩展到与 mean 相同的维度
        if std.dim() == 1 and mean.dim() > 1:
            self.std = std.unsqueeze(0).expand_as(mean)
        else:
            self.std = std
        
        # 限制 std 的最小值，防止标准差太小或太大
        self.std = torch.clamp(self.std, min=1e-6)
        
        # 创建 PyTorch 分布
        self.dist = D.Normal(self.mean, self.std)
    
    def sample(self) -> torch.Tensor:
        """采样动作"""
        return self.dist.sample()
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """计算对数概率 - 对每个维度求和"""
        return self.dist.log_prob(actions).sum(dim=-1)
    
    def entropy(self) -> torch.Tensor:
        """计算熵 - 对每个维度求和"""
        return self.dist.entropy().sum(dim=-1)
    
    def kl(self, other: 'Continuous') -> torch.Tensor:
        """计算 KL 散度 - 对每个维度求和"""
        # 使用独立高斯分布的 KL 散度公式
        kl = torch.log(other.std / self.std) + \
             (self.std ** 2 + (self.mean - other.mean) ** 2) / (2 * other.std ** 2) - 0.5
        return kl.sum(dim=-1)
    
    def mode(self) -> torch.Tensor:
        """返回均值"""
        return self.mean
