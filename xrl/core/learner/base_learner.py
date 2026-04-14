"""Learner 基类接口"""

from typing import Dict, Any, Optional
from xrl.interfaces.model import Model
from xrl.core.replay_buffer.base_replay_buffer import BaseReplayBuffer


class BaseLearner:
    """Learner 基类"""
    
    def __init__(self, models: Dict[str, Model], replay_buffer: BaseReplayBuffer, config: Optional[Dict[str, Any]] = None):
        """初始化 Learner
        
        Args:
            models: 模型实例字典
            replay_buffer: Replay Buffer 实例（Ray Actor 句柄）
            config: 配置字典
        """
        self.models = models
        self.replay_buffer = replay_buffer
        self.config = config or {}
        
        learner_config = self.config.get("learner", {})
        self.batch_size = learner_config.get("batch_size", 512)  # 更大的批量大小
        self.epochs = learner_config.get("epochs", 4)  # 训练轮数
    
    def get_models(self) -> Dict[str, Model]:
        """获取所有模型
        
        Returns:
            模型字典
        """
        return self.models
    
    def train(self) -> None:
        """训练模型
        
        通用训练逻辑：从 Replay Buffer 采样数据，然后调用每个模型的 learn 方法进行训练
        """
        raise NotImplementedError
    
    def update_parameters(self) -> None:
        """更新模型参数并同步给 Predictor"""
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """保存模型"""
        for model_name, model in self.models.items():
            model.save(f"{path}/{model_name}.pt")
    
    def load(self, path: str) -> None:
        """加载模型"""
        for model_name, model in self.models.items():
            model.load(f"{path}/{model_name}.pt")
    
    def train_from_league(self, model_name: str) -> None:
        """基于联赛结果训练模型
        
        Args:
            model_name: 模型名称
        """
        # 基于联赛结果的训练逻辑
        pass
    
    def should_train(self) -> bool:
        """判断是否应该训练
        
        Returns:
            是否应该训练
        """
        return True
    
    def adjust_batch_size(self) -> int:
        """调整批量大小
        
        Returns:
            调整后的批量大小
        """
        return self.batch_size
    
    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """获取模型参数
        
        Args:
            model_name: 模型名称
        
        Returns:
            模型参数
        """
        if model_name in self.models and hasattr(self.models[model_name], 'get_parameters'):
            return self.models[model_name].get_parameters()
        return {}