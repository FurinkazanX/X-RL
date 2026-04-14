"""Pendulum 智能体实现"""

import numpy as np
from xrl.interfaces.agent import Agent
from typing import Dict, Any


class PendulumAgent(Agent):
    """Pendulum 智能体实现"""
    
    def __init__(self, models: Dict[str, Any], predictors: Dict[str, Any] = None):
        """初始化智能体
        
        Args:
            models: 模型实例字典，键为模型名称，值为模型实例
            predictors: 预测器实例字典，键为模型名称，值为预测器实例
        """
        super().__init__(models, predictors)
        # 获取第一个模型（通常只有一个模型）
        self.model_name = list(models.keys())[0] if models else None
        self.model = models.get(self.model_name) if self.model_name else None
        self.predictor = predictors.get(self.model_name) if self.model_name and predictors else None
    
    def step(self, obs: Dict[str, Any]) -> tuple:
        """根据状态输出仿真指令，内部会根据是否存在 Predictor 选择预测方式
        
        Args:
            obs: 环境观测
        
        Returns:
            包含动作和 StepInfo 对象的元组
        """
        if not self.model:
            raise ValueError("Model not initialized")
        
        # 使用预测器或直接使用模型进行预测
        if self.predictor:
            # 使用远程预测器
            result = self.predictor.predict.remote(self.model_name, {"state": obs, "model_name": self.model_name})
            actions = result["actions"]
            action = actions["action"]  # 从 actions 字典中提取单个动作
            # 从结果中提取 StepInfo 对象
            step_info = result.get("step_info", None)
            if not step_info:
                # 如果没有 StepInfo 对象，创建一个
                from xrl.algo.ppo.ppo_step_info import PPOStepInfo
                step_info = PPOStepInfo(state=obs, model_output=result, model_name=self.model_name)
        else:
            # 直接使用本地模型，调用 forward(train=False)
            result = self.model.forward({"state": obs, "model_name": self.model_name}, train=False)
            actions = result["actions"]
            action = actions["action"]  # 从 actions 字典中提取单个动作
            step_info = result["step_info"]
        
        return (action, step_info)
    
    def reset(self) -> None:
        """重置 Agent 状态"""
        # 这里可以实现重置逻辑
        super().reset()