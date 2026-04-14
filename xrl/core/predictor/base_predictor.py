"""Predictor 基类接口"""

from typing import Dict, Any, Optional
from xrl.interfaces.model import Model


class BasePredictor:
    """Predictor 基类"""
    
    def __init__(self, models: Dict[str, Model]):
        """初始化 Predictor
        
        Args:
            models: 模型实例字典
        """
        self.models = models
    
    def predict(self, model_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """预测接口
        
        Args:
            model_name: 模型名称
            state: 状态数据
        
        Returns:
            预测结果
        """
        raise NotImplementedError
    
    def update_parameters(self, model_name: str, parameters: Dict[str, Any]) -> None:
        """更新模型参数
        
        Args:
            model_name: 模型名称
            parameters: 模型参数
        """
        raise NotImplementedError
