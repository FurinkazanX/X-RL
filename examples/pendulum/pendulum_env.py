"""Pendulum 环境实现"""

import gymnasium as gym
from xrl.interfaces.env import Env
from typing import Dict, Any, Tuple


class PendulumEnv(Env):
    """Pendulum 环境实现"""
    
    def __init__(self, render_mode=None):
        """初始化环境
        
        Args:
            render_mode: 渲染模式
        """
        self.env = gym.make("Pendulum-v1", render_mode=render_mode)
        self.episode_reward = 0
        self.current_episode = 0
    
    def reset(self) -> Dict[str, Any]:
        """重置环境，返回初始状态
        
        Returns:
            初始状态
        """
        # 重置 reward 计数器
        self.episode_reward = 0
        self.current_episode += 1
        
        state, _ = self.env.reset()
        return state
    
    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """执行动作，返回下一个状态、奖励、是否结束、额外信息
        
        Args:
            actions: 智能体动作
        
        Returns:
            下一个状态、奖励、是否结束、额外信息
        """
        # 假设只有一个智能体
        action = actions.get("agent", 0)
        next_state, reward, done, truncated, info = self.env.step(action)     
        self.episode_reward += reward
        
        rewards = {"agent": reward}
        dones = {"agent": done or truncated}
        
        if done or truncated:
            info["episode_reward"] = self.episode_reward
            info["episode_count"] = self.current_episode
            print(f"Episode {self.current_episode} reward: {self.episode_reward}")
        
        return next_state, rewards, dones, info
    
    def close(self) -> None:
        """关闭环境"""
        self.env.close()
