"""本地 Predictor 实现"""

import ray
from typing import Dict, Any
from xrl.core.predictor.base_predictor import BasePredictor


@ray.remote
class LocalPredictor(BasePredictor):
    """本地 Predictor 实现"""
    
    def predict(self, model_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """预测接口
        
        Args:
            model_name: 模型名称
            state: 状态数据
        
        Returns:
            预测结果
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        result = self.models[model_name].forward(state)
        
        # 检查结果类型
        if isinstance(result, tuple) and len(result) == 2:
            # 模型返回的是 (action, step_info) 元组
            action, step_info = result
            # 提取模型输出信息
            model_output = step_info.model_output if hasattr(step_info, 'model_output') else {}
            # 构造返回字典
            return {
                "action": action,
                "step_info": step_info,
                **model_output
            }
        else:
            # 保持向后兼容，处理返回字典的情况
            return result
    
    def update_parameters(self, model_name: str, parameters: Dict[str, Any]) -> None:
        """更新模型参数
        
        Args:
            model_name: 模型名称
            parameters: 模型参数
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        self.models[model_name].set_parameters(parameters)
