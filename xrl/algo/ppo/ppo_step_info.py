"""PPO StepInfo 类"""

import numpy as np
from xrl.core.types import StepInfo


class PPOStepInfo(StepInfo):
    """PPO Step 信息"""
    
    def __init__(self, state: any, model_output: dict, model_name: str):
        """初始化 PPO Step 信息
        
        Args:
            state: 当前状态
            model_output: 模型输出信息
            model_name: 模型名称
        """
        super().__init__(state, model_output, model_name)
        self.log_prob = None
        self.value = None
        self.dist_params = {}  # 保存分布参数，而非完整对象
        
        if "probs" in model_output:
            self.log_prob = np.log(model_output["probs"][model_output["action"]])
        if "value" in model_output:
            self.value = model_output["value"]
        
        if "dist_params" in model_output:
            self.dist_params = model_output["dist_params"]
        
        if "dist_params" in model_output and "log_prob" in model_output:
            self.log_prob = model_output["log_prob"]

        self.advantage = None
        self.return_ = None
