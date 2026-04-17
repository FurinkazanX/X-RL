"""Actor 基类接口"""

from typing import Dict, Any, Optional
from xrl.interfaces.env import Env
from xrl.interfaces.agent import Agent
from xrl.core.replay_buffer.base_replay_buffer import BaseReplayBuffer


class BaseActor:
    """Actor 基类"""

    def __init__(self, env: Env, agents: Dict[str, Agent], replay_buffer: BaseReplayBuffer, models: Optional[Dict[str, Any]] = None, gamma: float = 0.99, lam: float = 0.95):
        """初始化 Actor

        Args:
            env: 环境实例
            agents: Agent 实例字典
            replay_buffer: Replay Buffer 实例（Ray Actor 句柄）
            models: 模型实例字典
            gamma: 折扣因子
            lam: GAE lambda 参数
        """
        self.env = env
        self.agents = agents
        self.replay_buffer = replay_buffer
        self.models = models or {}
        self.gamma = gamma
        self.lam = lam
        # Ray 序列化会将 agents 和 models 各自独立反序列化，导致 agent 内部的
        # 模型引用与 self.models 指向不同对象。在此重新绑定，确保 Agent 使用
        # self.models 中的同一份实例，后续只需更新 self.models 即可。
        self._rebind_agent_model_references()

    def _rebind_agent_model_references(self) -> None:
        """将所有 Agent 内部的模型引用重新绑定到 self.models 中的对象"""
        for agent in self.agents.values():
            if hasattr(agent, 'models'):
                for model_name in list(agent.models.keys()):
                    if model_name in self.models:
                        agent.models[model_name] = self.models[model_name]
            if hasattr(agent, 'model') and hasattr(agent, 'model_name'):
                if agent.model_name in self.models:
                    agent.model = self.models[agent.model_name]
    
    def run(self) -> None:
        """运行 Actor，与环境交互并收集经验"""
        raise NotImplementedError
    
    def reset(self) -> None:
        """重置 Actor 状态"""
        raise NotImplementedError
    
    def update_model_parameters(self, model_name: str, parameters: Dict[str, Any]) -> None:
        """更新模型参数，并同步到 Agent 持有的 Predictor

        Args:
            model_name: 模型名称
            parameters: 模型参数
        """
        if model_name in self.models:
            self.models[model_name].set_parameters(parameters)

        # Agent 持有 Predictor 的 Ray Actor 句柄，顺带推送参数更新，
        # Controller 无需单独感知 Predictor 的存在
        for agent in self.agents.values():
            if model_name in agent.predictors:
                agent.predictors[model_name].update_parameters.remote(model_name, parameters)
