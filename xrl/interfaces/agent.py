"""Agent 接口定义"""

from typing import Dict, Any, Optional
from xrl.core.types import Experience


class Agent:
    """智能体接口"""
    
    def __init__(self, models: Dict[str, Any], predictors: Dict[str, Any] = None):
        """初始化智能体
        
        Args:
            models: 模型实例字典，键为模型名称，值为模型实例
            predictors: 预测器实例字典，键为模型名称，值为预测器实例
        """
        self.models = models
        self.predictors = predictors or {}
    
    def step(self, obs: Dict[str, Any]) -> tuple:
        """根据状态输出仿真指令，内部会根据是否存在 Predictor 选择预测方式
        
        Args:
            obs: 环境观测
        
        Returns:
            包含动作和 StepInfo 对象的元组，格式为 (action, step_info)
        """
        raise NotImplementedError
    
    def reset(self) -> None:
        """重置 Agent 状态"""
        pass
