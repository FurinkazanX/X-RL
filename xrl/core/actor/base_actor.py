"""Actor 基类接口"""

from typing import Dict, Any, Optional
from xrl.interfaces.env import Env
from xrl.interfaces.agent import Agent
from xrl.core.replay_buffer.base_replay_buffer import BaseReplayBuffer


class BaseActor:
    """Actor 基类"""
    
    def __init__(self, env: Env, agents: Dict[str, Agent], replay_buffer: BaseReplayBuffer, models: Optional[Dict[str, Any]] = None):
        """初始化 Actor
        
        Args:
            env: 环境实例
            agents: Agent 实例字典
            replay_buffer: Replay Buffer 实例（Ray Actor 句柄）
            models: 模型实例字典
        """
        self.env = env
        self.agents = agents
        self.replay_buffer = replay_buffer
        self.models = models or {}
    
    def run(self) -> None:
        """运行 Actor，与环境交互并收集经验"""
        raise NotImplementedError
    
    def reset(self) -> None:
        """重置 Actor 状态"""
        raise NotImplementedError
    
    def update_model_parameters(self, model_name: str, parameters: Dict[str, Any]) -> None:
        """更新模型参数
        
        Args:
            model_name: 模型名称
            parameters: 模型参数
        """
        if model_name in self.models:
            self.models[model_name].set_parameters(parameters)
        
        # 同时更新 Agent 中的模型参数
        for agent in self.agents.values():
            if hasattr(agent, 'models') and model_name in agent.models:
                agent.models[model_name].set_parameters(parameters)
