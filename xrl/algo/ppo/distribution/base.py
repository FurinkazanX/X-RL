"""概率分布基类"""

import torch
from abc import ABC, abstractmethod
from typing import Union, Optional


class Distribution(ABC):
    """概率分布基类"""
    
    @abstractmethod
    def sample(self) -> torch.Tensor:
        """采样动作
        
        Returns:
            采样到的动作
        """
        pass
    
    @abstractmethod
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """计算给定动作的对数概率
        
        Args:
            actions: 需要计算的动作
            
        Returns:
            对数概率
        """
        pass
    
    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """计算分布的熵
        
        Returns:
            熵
        """
        pass
    
    @abstractmethod
    def kl(self, other: 'Distribution') -> torch.Tensor:
        """计算当前分布与另一个分布的 KL 散度
        
        Args:
            other: 另一个分布
            
        Returns:
            KL 散度
        """
        pass
    
    def mode(self) -> torch.Tensor:
        """返回分布的众数（最可能的动作）
        
        Returns:
            众数
        """
        raise NotImplementedError("mode() is not implemented for this distribution")
