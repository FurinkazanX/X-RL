"""核心数据类型定义"""

from typing import Dict, List, Any, Optional, Tuple


class Experience:
    """经验数据类"""
    def __init__(self, state, action, reward, next_state, done, info=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info or {}


class Trajectory:
    """轨迹数据类"""
    def __init__(self, experiences: List):
        self.experiences = experiences
    
    def __len__(self):
        return len(self.experiences)


class Batch:
    """批量数据类"""
    def __init__(self, states, actions, rewards, next_states, dones, infos=None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.infos = infos or {}


class StepInfo(Experience):
    """Step 信息基类，继承自 Experience"""
    
    def __init__(self, state: Any, model_output: Dict[str, Any], model_name: str):
        """初始化 Step 信息
        
        Args:
            state: 当前状态
            model_output: 模型输出信息
            model_name: 模型名称
        """
        # 先不设置 reward, action, next_state, done，这些在 update 方法中设置
        super().__init__(
            state=state,
            action=None,
            reward=0.0,
            next_state=None,
            done=False,
            info={"model_output": model_output, "model_name": model_name}
        )
        self.model_output = model_output
        self.model_name = model_name
    
    def update(self, reward: float, done: bool, next_state: Any, action: Any, info: Optional[Dict[str, Any]] = None) -> None:
        """更新 Step 信息
        
        Args:
            reward: 奖励
            done: 是否结束
            next_state: 下一个状态
            action: 动作
            info: 环境返回的额外信息
        """
        self.reward = reward
        self.done = done
        self.next_state = next_state
        self.action = action
        
        if info:
            # 把环境返回的 info 合并到 step_info 的 info 中
            self.info.update(info)
