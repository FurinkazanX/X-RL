# X-RL: 基于 Ray 的分布式强化学习框架

## 1. 项目概述

X-RL 是一个基于 Ray 的分布式强化学习框架，旨在提供高效、可扩展的强化学习训练环境。框架支持本地训练和跨机器的分布式训练，通过配置文件灵活定义不同组件的运行节点。

## 2. 核心组件

### 2.1 四大核心要素

- **Actor**: 负责与环境交互，收集经验数据
- **Learner**: 负责从经验数据中学习，更新模型参数
- **Predictor**: 负责提供模型预测，为 Actor 提供行动建议
- **Replay Buffer**: 负责存储和管理经验数据，支持异步训练时的数据交换

### 2.2 用户开发接口

- **Env**: 环境接口，用户需要实现的环境交互逻辑
- **Agent**: 智能体接口，定义智能体的行为策略
- **Model**: 模型接口，定义神经网络模型结构

## 3. 目录结构

```
X-RL/
├── xrl/
│   ├── algo/
│   │   ├── ppo/
│   │   │   ├── model.py       # PPO 模型基类
│   │   │   └── step_info.py   # PPO StepInfo 类
│   │   ├── dqn/
│   │   │   ├── model.py       # DQN 模型基类
│   │   │   └── step_info.py   # DQN StepInfo 类
│   │   └── sac/
│   │       ├── model.py       # SAC 模型基类
│   │       └── step_info.py   # SAC StepInfo 类
│   ├── core/
│   │   ├── actor/
│   │   │   ├── __init__.py   # Actor 模块初始化
│   │   │   ├── base.py       # Actor 基类接口
│   │   │   └── actor.py      # Actor 实现
│   │   ├── learner/
│   │   │   ├── __init__.py   # Learner 模块初始化
│   │   │   ├── base.py       # Learner 基类接口
│   │   │   └── learner.py    # 通用 Learner 实现
│   │   ├── predictor/
│   │   │   ├── __init__.py   # Predictor 模块初始化
│   │   │   ├── base.py       # Predictor 基类接口
│   │   │   ├── local.py      # 本地 Predictor 实现
│   │   │   └── distributed.py # 分布式 Predictor 实现
│   │   ├── replay_buffer/
│   │   │   ├── __init__.py   # Replay Buffer 模块初始化
│   │   │   ├── base.py       # Replay Buffer 基类接口
│   │   │   ├── uniform.py    # 均匀采样 Replay Buffer 实现
│   │   │   └── prioritized.py # 优先级采样 Replay Buffer 实现
│   │   ├── experience_processor/
│   │   │   ├── __init__.py   # Experience Processor 模块初始化
│   │   │   ├── base.py       # Experience Processor 基类接口
│   │   │   ├── dqn.py        # DQN 经验处理器实现
│   │   │   └── ppo.py        # PPO 经验处理器实现
│   │   └── league/
│   │       ├── __init__.py   # 联赛模块初始化
│   │       ├── base.py       # 联赛基类接口
│   │       ├── manager.py    # 联赛管理器实现
│   │       ├── evaluator.py  # 评估器实现
│   │       └── selector.py   # 智能体选择器实现
│   ├── types.py        # 核心数据类型定义
│   ├── interfaces/
│   │   ├── env.py          # Env 接口定义
│   │   ├── agent.py        # Agent 接口定义
│   │   └── model.py        # Model 接口定义
│   ├── controllers/
│   │   ├── __init__.py     # Controller 模块初始化
│   │   ├── base.py         # Controller 基类接口
│   │   └── default.py      # 默认 Controller 实现
│   ├── summary/
│   │   ├── __init__.py     # Summary 模块初始化
│   │   ├── base.py         # Summary 基类接口
│   │   ├── tensorboard.py  # TensorBoard 实现
│   │   └── wandb.py        # Weights & Biases 实现
│   ├── utils/
│   │   ├── config.py       # 配置文件处理
│   │   ├── logger.py       # 日志工具
│   │   └── ray_utils.py    # Ray 相关工具函数
│   └── main.py             # 主入口文件
├── configs/
│   ├── local.yaml          # 本地训练配置
│   └── distributed.yaml    # 分布式训练配置
├── examples/
│   ├── cartpole/
│   │   ├── env.py          # CartPole 环境实现
│   │   ├── agent.py        # CartPole 智能体实现
│   │   ├── model.py        # CartPole 模型实现
│   │   └── config.yaml     # CartPole 配置文件
│   └── pendulum/
│       ├── env.py          # Pendulum 环境实现
│       ├── agent.py        # Pendulum 智能体实现
│       ├── model.py        # Pendulum 模型实现
│       └── config.yaml     # Pendulum 配置文件
├── setup.py                # 安装脚本
├── requirements.txt        # 依赖项
└── README.md               # 项目说明
```

## 4. 核心组件设计

### 4.1 数据结构

**功能**：
- 定义经验数据和批量数据的结构
- 支持不同算法对数据的不同要求
- 提供统一的基类接口，便于扩展

**基类接口**：

```python
class Experience:
    """经验数据基类"""
    
    def __init__(self, state: Any, action: Any, reward: float, next_state: Any, done: bool, info: Optional[Dict[str, Any]] = None):
        """初始化经验数据
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
            info: 额外信息
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "info": self.info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """从字典创建"""
        return cls(
            state=data["state"],
            action=data["action"],
            reward=data["reward"],
            next_state=data["next_state"],
            done=data["done"],
            info=data.get("info", {})
        )


class Trajectory:
    """轨迹数据基类"""
    
    def __init__(self, experiences: List[Experience]):
        """初始化轨迹数据
        
        Args:
            experiences: 经验数据列表
        """
        self.experiences = experiences
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "experiences": [exp.to_dict() for exp in self.experiences]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """从字典创建"""
        experiences = [Experience.from_dict(exp) for exp in data["experiences"]]
        return cls(experiences)
```

**算法特定数据结构**：

```python
class PPOExperience(Experience):
    """PPO 经验数据"""
    
    def __init__(self, state: Any, action: Any, reward: float, next_state: Any, done: bool, log_prob: float, value: float, advantage: Optional[float] = None, info: Optional[Dict[str, Any]] = None):
        """初始化 PPO 经验数据
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
            log_prob: 动作的对数概率
            value: 状态价值
            advantage: 优势函数值
            info: 额外信息
        """
        super().__init__(state, action, reward, next_state, done, info)
        self.log_prob = log_prob
        self.value = value
        self.advantage = advantage


class DQNExperience(Experience):
    """DQN 经验数据"""
    pass


class PPOBatch:
    """PPO 批量数据"""
    
    def __init__(self, states: List[Any], actions: List[Any], rewards: List[float], next_states: List[Any], dones: List[bool], log_probs: List[float], values: List[float], advantages: List[float]):
        """初始化 PPO 批量数据
        
        Args:
            states: 状态列表
            actions: 动作列表
            rewards: 奖励列表
            next_states: 下一个状态列表
            dones: 是否结束列表
            log_probs: 动作的对数概率列表
            values: 状态价值列表
            advantages: 优势函数值列表
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.log_probs = log_probs
        self.values = values
        self.advantages = advantages


class DQNBatch:
    """DQN 批量数据"""
    
    def __init__(self, states: List[Any], actions: List[Any], rewards: List[float], next_states: List[Any], dones: List[bool]):
        """初始化 DQN 批量数据
        
        Args:
            states: 状态列表
            actions: 动作列表
            rewards: 奖励列表
            next_states: 下一个状态列表
            dones: 是否结束列表
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones


class GenericBatch:
    """通用批量数据"""
    
    def __init__(self, states: List[Any], actions: List[Any], rewards: List[float], next_states: List[Any], dones: List[bool], infos: List[Dict[str, Any]]):
        """初始化通用批量数据
        
        Args:
            states: 状态列表
            actions: 动作列表
            rewards: 奖励列表
            next_states: 下一个状态列表
            dones: 是否结束列表
            infos: 额外信息列表
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.infos = infos


class StepInfo:
    """Step 信息基类"""
    
    def __init__(self, state: Any, model_output: Dict[str, Any], model_name: str):
        """初始化 Step 信息
        
        Args:
            state: 当前状态
            model_output: 模型输出信息
            model_name: 模型名称
        """
        self.state = state
        self.model_output = model_output
        self.model_name = model_name
    
    def update(self, reward: float, done: bool, next_state: Any, action: Any) -> None:
        """更新 Step 信息
        
        Args:
            reward: 奖励
            done: 是否结束
            next_state: 下一个状态
            action: 动作
        """
        self.reward = reward
        self.done = done
        self.next_state = next_state
        self.action = action


class PPOStepInfo(StepInfo):
    """PPO Step 信息"""
    
    def __init__(self, state: Any, model_output: Dict[str, Any], model_name: str):
        """初始化 PPO Step 信息
        
        Args:
            state: 当前状态
            model_output: 模型输出信息
            model_name: 模型名称
        """
        super().__init__(state, model_output, model_name)
        self.log_prob = None
        self.value = None
        
        if "probs" in model_output:
            self.log_prob = np.log(model_output["probs"][model_output["action"]])
        if "value" in model_output:
            self.value = model_output["value"]


class DQNStepInfo(StepInfo):
    """DQN Step 信息"""
    pass


class SACStepInfo(StepInfo):
    """SAC Step 信息"""
    pass
```

**数据流程**：
1. **智能体执行动作**：Agent 在 `step` 方法中执行动作，返回包含动作和 StepInfo 对象的元组
2. **Actor 收集信息**：Actor 收集所有 Agent 的动作和 StepInfo 对象
3. **执行环境步骤**：Actor 执行环境步骤，获得 reward、done 等信息
4. **更新信息**：Actor 调用 StepInfo 的 `update` 方法，更新 reward、done、next_state 等信息
5. **构建轨迹**：Actor 将更新后的 StepInfo 对象添加到轨迹中
6. **创建经验数据**：Actor 从轨迹中的 StepInfo 对象创建经验数据对象
7. **经验处理器后处理**：Experience Processor 对轨迹进行后处理，计算算法特定的额外信息（如 Advantage）
8. **数据存储**：将处理后的经验数据存储到 Replay Buffer
9. **Learner 采样数据**：Learner 从 Replay Buffer 采样经验数据，创建批量数据
10. **模型训练**：Learner 将批量数据传给 Model 的 `learn` 方法进行训练

### 4.2 Actor

**功能**：
- 与环境交互，执行动作
- 收集经验数据
- 将经验数据发送给 Replay Buffer
- 支持多智能体场景，一个 Actor 可以管理多个 Agent
- 基于 Ray 实现，支持分布式部署

**基类接口**：

```python
import ray

@ray.remote
class BaseActor:
    def __init__(self, env, agents, replay_buffer, experience_processor=None):
        """初始化 Actor
        
        Args:
            env: 环境实例
            agents: Agent 实例字典
            replay_buffer: Replay Buffer 实例（Ray Actor 句柄）
            experience_processor: 经验处理器实例
        """
        self.env = env
        self.agents = agents
        self.replay_buffer = replay_buffer
        self.experience_processor = experience_processor
    
    def run(self):
        """运行 Actor，与环境交互并收集经验"""
        pass
    
    def reset(self):
        """重置 Actor 状态"""
        pass
```

**实现细节**：
- 使用 Ray 的 Actor 模式实现
- 支持并行运行多个 Actor 实例
- 可配置的经验收集策略
- 使用 Agent 的 step 方法获取动作
- 在每个 episode 开始时调用 Agent 的 reset 方法
- 不直接与 Predictor 交互，而是通过 Agent 间接获取预测服务
- 支持多智能体场景，如足球比赛中每个 Agent 控制一个球员
- 为每个 Agent 维护独立的状态和经验数据
- 多Agent之间的通信协作不需要actor负责，Agent可以支持嵌套模式，并且Agent支持多model，由Agent内部进行通信协作管理
- 统一的 Actor 实现，通过传入的 Agent 数量自动支持单智能体和多智能体场景
- 支持经验处理器机制，实现与具体算法的解耦
- 经验处理器负责处理完整轨迹，计算算法特定的额外信息（如 Advantage）

### 4.3 Model

**功能**：
- 定义神经网络模型结构
- 实现前向传播，同时返回策略和价值网络的输出
- 支持模型参数的获取和设置
- 支持模型的保存和加载

**基类接口**：

```python
from typing import Dict, Any

class Model:
    """模型接口"""
    
    def __init__(self, **kwargs):
        """初始化模型
        
        Args:
            **kwargs: 模型参数
        """
        pass
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """前向传播，同时返回策略和价值网络的输出
        
        Args:
            inputs: 输入数据
        
        Returns:
            输出数据，包含策略和价值网络的输出
        """
        raise NotImplementedError
    
    def learn(self, batch: Any) -> None:
        """更新模型参数
        
        Args:
            batch: 批量数据，具体类型由算法决定
        """
        raise NotImplementedError
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数
        
        Returns:
            模型参数
        """
        raise NotImplementedError
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """设置模型参数
        
        Args:
            parameters: 模型参数
        """
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """保存模型
        
        Args:
            path: 保存路径
        """
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """加载模型
        
        Args:
            path: 加载路径
        """
        raise NotImplementedError
```

**实现细节**：
- 一个 Model 对象包含完整的强化学习神经网络模型，如 PPO 中的策略网络和价值网络
- 在 forward 方法中同时返回策略和价值网络的输出
- 支持模型参数的获取和设置，便于与 Predictor 同步
- 支持模型的保存和加载，便于断点续训和模型部署
- 具体的强化学习算法训练逻辑在算法目录中的模型基类中实现，如 PPOModel、DQNModel 等

### 4.4 算法实现

**功能**：
- 实现不同强化学习算法的核心逻辑
- 为每种算法提供专门的模型基类和 StepInfo 类
- 封装算法特定的训练逻辑，便于用户继承和扩展

**目录结构**：
- `xrl/algo/ppo/`：PPO 算法实现
  - `model.py`：PPO 模型基类，实现 PPO 算法的训练逻辑
  - `step_info.py`：PPO StepInfo 类，存储 PPO 算法所需的额外信息
- `xrl/algo/dqn/`：DQN 算法实现
  - `model.py`：DQN 模型基类，实现 DQN 算法的训练逻辑
  - `step_info.py`：DQN StepInfo 类，存储 DQN 算法所需的额外信息
- `xrl/algo/sac/`：SAC 算法实现
  - `model.py`：SAC 模型基类，实现 SAC 算法的训练逻辑
  - `step_info.py`：SAC StepInfo 类，存储 SAC 算法所需的额外信息

**使用方式**：
用户可以继承对应算法的模型基类（如 `PPOModel`）来实现自己的神经网络，而不需要重新实现算法的训练逻辑。例如：

```python
from xrl.algo.ppo.model import PPOModel

class MyPPOModel(PPOModel):
    def __init__(self, input_dim, action_dim, hidden_dims):
        super().__init__()
        # 初始化神经网络结构
    
    def forward(self, inputs):
        # 实现前向传播
        pass
```

### 4.5 Learner

**功能**：
- 从 Replay Buffer 接收经验数据
- 批量处理经验数据
- 更新多个模型参数
- 将更新后的模型参数发送给对应的 Predictor
- 基于 Ray 实现，支持分布式部署

**基类接口**：

```python
from typing import Dict, Any, Optional
from xrl.interfaces.model import Model
from xrl.core.replay_buffer.base import BaseReplayBuffer

class BaseLearner:
    """Learner 基类"""
    
    def __init__(self, models: Dict[str, Model], replay_buffer: BaseReplayBuffer):
        """初始化 Learner
        
        Args:
            models: 模型实例字典
            replay_buffer: Replay Buffer 实例（Ray Actor 句柄）
        """
        self.models = models
        self.replay_buffer = replay_buffer
        self.batch_size = 256  # 默认批量大小
        self.epochs = 4  # 默认训练轮数
    
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
```

**实现细节**：
- 使用 Ray 的 Actor 模式实现
- 支持通用 Learner，不局限于特定算法
- 可配置的学习率和批量大小
- 使用 Model 的 learn 方法进行模型更新，具体算法实现放在 Model 中
- 支持多个 Model 的并行训练和参数更新
- 提供 Learner 实现，适用于各种强化学习算法

**通用 Learner**：
- 不局限于某个特定算法，而是调用模型的 learn 方法来执行具体的强化学习算法训练
- 从 Replay Buffer 采样数据，然后调用每个模型的 learn 方法进行训练
- 支持多种强化学习算法，具体实现由 Model 负责

### 4.6 Predictor

**功能**：
- 接收 Learner 发送的多个模型参数
- 为 Agent 提供多个模型的行动预测
- 支持批量预测（batch predict）
- 实现在 GPU 服务器上远程计算，将结果返回到 Agent 所在节点
- 支持多层 Predictor 架构，实现负载均衡
- 基于 Ray 实现，支持分布式部署

**基类接口**：

```python
import ray

@ray.remote
class BasePredictor:
    def __init__(self, models):
        """初始化 Predictor
        
        Args:
            models: 模型实例字典
        """
        self.models = models
    
    def predict(self, model_name, state):
        """预测接口
        
        Args:
            model_name: 模型名称
            state: 状态数据
        """
        pass
    
    def update_parameters(self, model_name, parameters):
        """更新模型参数
        
        Args:
            model_name: 模型名称
            parameters: 模型参数
        """
        pass
```

**实现细节**：
- 使用 Ray 的 Actor 模式实现，支持跨节点部署
- 支持多个模型的缓存和异步更新
- 可配置的预测策略
- 实现 batch predict 功能，收集到一个 batch 大小的 state 数据后一起做预测，提高计算效率
- 支持 GPU 加速，可部署在 GPU 服务器上为 CPU 服务器上的 Agent 提供远程预测服务
- 通过异步接口与 Agent 交互，支持非阻塞式预测
- 实现多层 Predictor 架构，外层 Predictor 管理多个子 Predictor，实现负载均衡
- 提供多种 Predictor 实现，如本地 Predictor 和分布式 Predictor

**与 Model 的关系**：
- **初始化**：Predictor 初始化时接收多个 Model 实例或初始参数
- **参数同步**：Learner 更新 Model 参数后，将新参数发送给对应的 Predictor
- **预测调用**：Predictor 在执行预测时，使用内部维护的对应 Model 实例进行前向传播
- **可选性**：Agent 可以选择直接使用本地 Model 还是通过 Predictor 进行远程预测

**多层 Predictor 架构**：
- **外层 Predictor**：对用户暴露统一接口，内部管理多个子 Predictor
- **子 Predictor**：负责具体的预测任务，分布在不同的机器上
- **负载均衡**：外层 Predictor 根据负载情况智能分配预测任务到不同的子 Predictor
- **故障隔离**：子 Predictor 之间相互独立，一个子 Predictor 故障不会影响其他子 Predictor

**调用流程**：
1. Actor 获取环境状态并调用 Agent.step()
2. Agent 在 step() 方法中根据配置选择使用本地 Model 或远程 Predictor
3. 如果使用 Predictor，Predictor 收集多个 Agent 的预测请求，当达到 batch_size 或超时后执行批量预测
4. Predictor 将预测结果返回给 Agent
5. Agent 处理预测结果并生成动作返回给 Actor

### 4.7 Replay Buffer

**功能**：
- 存储 Actor 收集的经验数据
- 为 Learner 提供批量采样数据
- 支持经验数据的管理和维护
- 支持不同的采样策略（如均匀采样、优先级采样、先进先出、后进先出等）
- 支持数据重用，可以设置数据最大重用次数
- 基于 Ray 实现，支持分布式部署

**基类接口**：

```python
import ray

@ray.remote
class BaseReplayBuffer:
    def __init__(self, size):
        """初始化 Replay Buffer
        
        Args:
            size: 缓冲区大小
        """
        self.size = size
    
    def add(self, experience):
        """添加经验数据
        
        Args:
            experience: 经验数据
        """
        pass
    
    def sample(self, batch_size):
        """采样批量数据
        
        Args:
            batch_size: 批量大小
        
        Returns:
            批量经验数据
        """
        pass
    
    def size(self):
        """返回缓冲区当前大小"""
        pass
    
    def clear(self):
        """清空缓冲区"""
        pass
```

**实现细节**：
- 使用 Ray 的 Actor 模式实现，确保数据的并发访问安全
- 支持固定大小的缓冲区，自动替换旧数据
- 可配置的缓冲区大小和采样策略
- 支持经验数据的序列化和反序列化
- 提供多种 Replay Buffer 实现，如均匀采样 Replay Buffer 和优先级采样 Replay Buffer

### 4.8 Experience Processor

**功能**：
- 处理完整轨迹数据
- 计算算法特定的额外信息（如 Advantage）
- 实现 Actor 与具体算法的解耦
- 支持不同算法的特定处理需求
- 支持与 Ray 分布式环境集成

### 4.9 Controller

**功能**：
- 调度不同要素的分布式运行
- 管理 Actor、Learner、Predictor 和 Replay Buffer 的生命周期
- 处理配置文件，根据配置初始化和启动各组件
- 监控系统运行状态，处理异常情况
- 支持用户自定义 Controller 实现
- 管理 Summary 模块，记录训练过程指标
- 支持跨机器的分布式部署配置

**基类接口**：

```python
class BaseController:
    def __init__(self, config):
        """初始化 Controller
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.components = {}
    
    def initialize(self):
        """初始化所有组件"""
        pass
    
    def start(self):
        """启动所有组件"""
        pass
    
    def stop(self):
        """停止所有组件"""
        pass
    
    def monitor(self):
        """监控系统运行状态"""
        pass
```

**默认实现**：

```python
class DefaultController(BaseController):
    def initialize(self):
        """初始化所有组件"""
        # 初始化 Summary
        summary_config = self.config.get("summary", {})
        if summary_config.get("enabled", False):
            summary_cls = getattr(
                importlib.import_module("xrl.summary"),
                summary_config.get("type", "TensorBoardSummary")
            )
            self.components["summary"] = summary_cls(summary_config)
            self.components["summary"].add_config(self.config)
        
        # 初始化联赛管理器
        league_config = self.config.get("league", {})
        if league_config.get("enabled", False):
            league_cls = getattr(
                importlib.import_module("xrl.core.league"),
                league_config.get("type", "LeagueManager")
            )
            self.components["league"] = league_cls.remote(league_config)
        
        # 初始化 Replay Buffer
        replay_buffer_config = self.config.get("replay_buffer", {})
        replay_buffer_cls = getattr(
            importlib.import_module("xrl.core.replay_buffer"),
            replay_buffer_config.get("type", "UniformReplayBuffer")
        )
        self.components["replay_buffer"] = replay_buffer_cls.remote(
            replay_buffer_config.get("size", 1000000)
        )
        
        # 初始化 Models
        models_config = self.config.get("models", {})
        models = {}
        for model_name, model_config in models_config.items():
            model_cls = getattr(
                importlib.import_module(model_config["module"]),
                model_config["class"]
            )
            models[model_name] = model_cls(**model_config.get("params", {}))
        
        # 初始化 Predictors
        predictors_config = self.config.get("predictors", {})
        predictors = {}
        for predictor_name, predictor_config in predictors_config.items():
            predictor_cls = getattr(
                importlib.import_module("xrl.core.predictor"),
                predictor_config.get("type", "LocalPredictor")
            )
            # 选择需要的模型
            model_names = predictor_config.get("models", [model_name])
            predictor_models = {name: models[name] for name in model_names if name in models}
            predictors[predictor_name] = predictor_cls.remote(predictor_models)
        
        # 初始化 Agents
        agents_config = self.config.get("agents", {})
        agents = {}
        for agent_name, agent_config in agents_config.items():
            agent_cls = getattr(
                importlib.import_module(agent_config["module"]),
                agent_config["class"]
            )
            # 选择需要的模型和 Predictor
            model_names = agent_config.get("models", [model_name])
            agent_models = {name: models[name] for name in model_names if name in models}
            predictor_names = agent_config.get("predictors", [])
            agent_predictors = {name: predictors[name] for name in predictor_names if name in predictors}
            agents[agent_name] = agent_cls(agent_models, agent_predictors)
        
        # 初始化 Experience Processor
        experience_processor_config = self.config.get("experience_processor", {})
        if experience_processor_config:
            experience_processor_cls = getattr(
                importlib.import_module("xrl.core.experience_processor"),
                experience_processor_config.get("type", "ExperienceProcessor")
            )
            # 初始化价值估计器
            value_estimator = None
            if "value_estimator" in experience_processor_config:
                ve_config = experience_processor_config["value_estimator"]
                if ve_config["type"] == "model":
                    model_name = ve_config["model"]
                    if model_name in models:
                        from xrl.core.experience_processor.base import ModelValueEstimator
                        value_estimator = ModelValueEstimator(models[model_name])
                elif ve_config["type"] == "predictor":
                    predictor_name = ve_config["predictor"]
                    model_name = ve_config["model"]
                    if predictor_name in predictors and model_name in models:
                        from xrl.core.experience_processor.base import PredictorValueEstimator
                        value_estimator = PredictorValueEstimator(predictors[predictor_name], model_name)
            
            self.components["experience_processor"] = experience_processor_cls(
                **experience_processor_config.get("params", {}),
                value_estimator=value_estimator
            )
        
        # 初始化 Actor
        actor_config = self.config.get("actor", {})
        actor_cls = getattr(
            importlib.import_module("xrl.core.actor"),
            actor_config.get("type", "Actor")
        )
        # 初始化环境
        env_config = self.config.get("env", {})
        env_cls = getattr(
            importlib.import_module(env_config["module"]),
            env_config["class"]
        )
        env = env_cls(**env_config.get("params", {}))
        
        self.components["actor"] = actor_cls.remote(
            env,
            agents,
            self.components["replay_buffer"],
            self.components.get("experience_processor")
        )
        
        # 初始化 Learner
        learner_config = self.config.get("learner", {})
        learner_cls = getattr(
            importlib.import_module("xrl.core.learner"),
            learner_config.get("type", "DQNLearner")
        )
        self.components["learner"] = learner_cls.remote(
            models,
            self.components["replay_buffer"]
        )
    
    def start(self):
        """启动所有组件"""
        # 启动 Actor
        if "actor" in self.components:
            self.components["actor"].run.remote()
        
        # 启动 Learner
        if "learner" in self.components:
            self.components["learner"].train.remote()
        
        # 启动联赛管理器
        if "league" in self.components:
            self.components["league"].run_season.remote()
    
    def stop(self):
        """停止所有组件"""
        # 停止所有组件
        for component_name, component in self.components.items():
            if hasattr(component, "stop"):
                component.stop.remote()
        
        # 关闭 Summary
        if "summary" in self.components:
            self.components["summary"].close()
    
    def monitor(self):
        """监控系统运行状态"""
        # 监控逻辑
        pass
```

### 4.10 Summary

**功能**：
- 记录训练过程中的各种指标（如奖励、损失、学习率等）
- 支持不同的可视化工具（如 TensorBoard、Weights & Biases）
- 提供统一的接口，方便用户切换不同的可视化工具
- 支持保存和加载实验配置

**基类接口**：

```python
class BaseSummary:
    def __init__(self, config):
        """初始化 Summary
        
        Args:
            config: 配置对象
        """
        self.config = config
    
    def scalar(self, tag, value, step):
        """记录标量值
        
        Args:
            tag: 标签名称
            value: 标量值
            step: 训练步数
        """
        pass
    
    def histogram(self, tag, values, step):
        """记录直方图
        
        Args:
            tag: 标签名称
            values: 数据值
            step: 训练步数
        """
        pass
    
    def image(self, tag, img_tensor, step):
        """记录图像
        
        Args:
            tag: 标签名称
            img_tensor: 图像张量
            step: 训练步数
        """
        pass
    
    def text(self, tag, text_string, step):
        """记录文本
        
        Args:
            tag: 标签名称
            text_string: 文本内容
            step: 训练步数
        """
        pass
    
    def add_config(self, config):
        """添加配置信息
        
        Args:
            config: 配置对象
        """
        pass
    
    def close(self):
        """关闭 Summary"""
        pass
```

**实现示例**：

```python
class TensorBoardSummary(BaseSummary):
    """TensorBoard 实现"""
    
    def __init__(self, config):
        super().__init__(config)
        import tensorflow as tf
        self.writer = tf.summary.create_file_writer(
            config.get("log_dir", "./logs")
        )
    
    def scalar(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
    
    def histogram(self, tag, values, step):
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step)
            self.writer.flush()
    
    def image(self, tag, img_tensor, step):
        with self.writer.as_default():
            tf.summary.image(tag, img_tensor, step=step)
            self.writer.flush()
    
    def text(self, tag, text_string, step):
        with self.writer.as_default():
            tf.summary.text(tag, text_string, step=step)
            self.writer.flush()
    
    def add_config(self, config):
        with self.writer.as_default():
            tf.summary.text("config", str(config), step=0)
            self.writer.flush()
    
    def close(self):
        self.writer.close()

class WandBSummary(BaseSummary):
    """Weights & Biases 实现"""
    
    def __init__(self, config):
        super().__init__(config)
        import wandb
        self.run = wandb.init(
            project=config.get("project", "x-rl"),
            name=config.get("name", None),
            config=config
        )
    
    def scalar(self, tag, value, step):
        self.run.log({tag: value}, step=step)
    
    def histogram(self, tag, values, step):
        self.run.log({tag: wandb.Histogram(values)}, step=step)
    
    def image(self, tag, img_tensor, step):
        self.run.log({tag: wandb.Image(img_tensor)}, step=step)
    
    def text(self, tag, text_string, step):
        self.run.log({tag: text_string}, step=step)
    
    def add_config(self, config):
        self.run.config.update(config)
    
    def close(self):
        self.run.finish()
```

**实现细节**：
- 提供统一的 Summary 接口，方便用户切换不同的可视化工具
- 支持常见的可视化操作，如标量、直方图、图像和文本
- 支持保存实验配置，便于实验复现
- 在 Controller 中初始化和使用 Summary 模块
- 在 Learner 和 Actor 中记录训练指标

### 4.11 League

**功能**：
- 管理智能体池，组织智能体之间的比赛
- 评估智能体性能并排名
- 执行选择、淘汰和更新策略
- 支持自博弈训练方法，促进智能体进化
- 基于 Ray 实现，支持分布式部署

**基类接口**：

```python
import ray

@ray.remote
class BaseLeague:
    def __init__(self, config):
        """初始化联赛
        
        Args:
            config: 联赛配置
        """
        self.config = config
        self.agent_pool = {}
    
    def add_agent(self, agent_id, agent):
        """添加智能体到联赛
        
        Args:
            agent_id: 智能体唯一标识符
            agent: 智能体实例
        """
        pass
    
    def remove_agent(self, agent_id):
        """从联赛中移除智能体
        
        Args:
            agent_id: 智能体唯一标识符
        """
        pass
    
    def run_season(self):
        """运行一个赛季的比赛"""
        pass
    
    def evaluate_agents(self):
        """评估所有智能体的性能"""
        pass
    
    def select_agents(self):
        """选择表现优秀的智能体"""
        pass
    
    def update_agents(self):
        """更新智能体池，替换表现差的智能体"""
        pass
```

**实现细节**：
- 使用 Ray 的 Actor 模式实现，支持分布式部署
- 管理智能体池，包括添加、删除和更新智能体
- 组织智能体之间的比赛，收集比赛结果
- 评估智能体性能，执行选择和更新策略
- 支持交叉和变异操作，生成新的智能体
- 与现有的 Actor、Learner、Predictor 框架集成

## 6. 配置文件设计

### 6.1 YAML 配置结构

```yaml
# 全局配置
global:
  algorithm: PPO  # 强化学习算法
  total_steps: 1000000  # 总训练步数
  log_interval: 1000  # 日志间隔
  save_interval: 10000  # 保存间隔

# Summary 配置
summary:
  enabled: true  # 是否启用 Summary
  type: TensorBoardSummary  # Summary 类型，可选 TensorBoardSummary 或 WandBSummary
  log_dir: ./logs  # TensorBoard 日志目录
  project: x-rl  # Weights & Biases 项目名称
  name: experiment-1  # Weights & Biases 实验名称

# Ray 配置
ray:
  address: "auto"  # Ray 集群地址，"auto" 表示自动检测，分布式部署时设置为 head node 地址
  redis_password: ""  # Redis 密码（如果需要）
  num_cpus: 4  # CPU 数量
  num_gpus: 1  # GPU 数量

# 分布式部署配置
distributed:
  actor_nodes: ["node1", "node2"]  # Actor 运行节点
  learner_nodes: ["node3"]  # Learner 运行节点
  predictor_nodes: ["node3", "node4"]  # Predictor 运行节点（建议 GPU 节点）
  replay_buffer_nodes: ["node5"]  # Replay Buffer 运行节点
  league_nodes: ["node6"]  # 联赛管理器运行节点

# 联赛配置
league:
  enabled: true  # 是否启用联赛
  type: LeagueManager  # 联赛管理器类型
  matches_per_season: 100  # 每个赛季的比赛数量
  top_k: 5  # 保留表现前 k 的智能体
  mutation_rate: 0.1  # 变异率
  crossover_rate: 0.5  # 交叉率
  population_size: 20  # 智能体池大小

# Actor 配置
actor:
  count: 4  # Actor 数量
  batch_size: 64  # 每个 Actor 的批量大小
  max_episode_steps: 1000  # 每个 episode 的最大步数
  nodes:  # Actor 运行节点
    - localhost

# Learner 配置
learner:
  batch_size: 256  # 批量大小
  learning_rate: 0.0003  # 学习率
  nodes:  # Learner 运行节点
    - localhost

# Predictor 配置
predictor:
  enabled: true  # 是否启用 Predictor
  batch_size: 64  # 批量大小
  timeout: 0.1  # 超时时间（秒）
  sub_predictors:  # 子 Predictor 配置
    predictor_1:
      batch_size: 64
      nodes:
        - localhost

# Replay Buffer 配置
replay_buffer:
  size: 1000000  # 缓冲区大小
  batch_size: 256  # 批量大小
  sampling_strategy: uniform  # 采样策略，可选 uniform 或 prioritized
  nodes:  # Replay Buffer 运行节点
    - localhost

# Experience Processor 配置
experience_processor:
  type: PPOExperienceProcessor  # 经验处理器类型
  gamma: 0.99  # 折扣因子
  lambda: 0.95  # GAE 参数
  value_estimator:  # 价值估计器配置
    type: model  # 类型，可选 model 或 predictor
    model: value_model  # 价值模型名称

# 环境配置
env:
  module: examples.cartpole.env  # 环境模块路径
  class: CartPoleEnv  # 环境类名称
  params:  # 环境参数
    render_mode: None

# 模型配置
models:
  policy:
    module: examples.cartpole.model  # 模型模块路径
    class: CartPolePolicy  # 模型类名称
    params:  # 模型参数
      input_dim: 4
      output_dim: 2
      hidden_dims: [64, 64]
  value:
    module: examples.cartpole.model  # 模型模块路径
    class: CartPoleValue  # 模型类名称
    params:  # 模型参数
      input_dim: 4
      hidden_dims: [64, 64]

# 智能体配置
agents:
  agent_1:
    module: examples.cartpole.agent  # 智能体模块路径
    class: CartPoleAgent  # 智能体类名称
    models:  # 智能体使用的模型
      - policy
      - value
    predictors:  # 智能体使用的 Predictor
      - predictor_1

# Controller 配置
controller:
  type: DefaultController  # Controller 类型
```

**实现细节**：
- 提供统一的 Summary 接口，方便用户切换不同的可视化工具
- 支持常见的可视化操作，如标量、直方图、图像和文本
- 支持保存实验配置，便于实验复现
- 在 Controller 中初始化和使用 Summary 模块
- 在 Learner 和 Actor 中记录训练指标


## 5. 用户接口设计

### 5.1 Env 接口

```python
class Env:
    def reset(self):
        """重置环境，返回初始状态"""
        pass
    
    def step(self, commands):
        """执行动作，返回下一个状态、奖励、是否结束"""
        pass
    
    def close(self):
        """关闭环境"""
        pass
```

### 5.2 Agent 接口

```python
class Agent:
    def __init__(self, models, predictors=None):
        """初始化 Agent
        
        Args:
            models: 本地模型实例字典，键为模型名称，值为模型实例
            predictors: 可选的远程 Predictor 实例字典，键为模型名称，值为 Predictor 实例
        """
        self.models = models
        self.predictors = predictors or {}
    
    def step(self, obs):
        """根据状态输出仿真指令，内部会根据是否存在 Predictor 选择预测方式"""
        pass
    
    def reset(self):
        """重置 Agent 状态"""
        pass
```

### 5.3 Model 接口

```python
class Model:
    def forward(self, state):
        """前向传播，返回动作或价值"""
        pass
    
    def learn(self, loss):
        """更新模型参数"""
        pass
    
    def get_parameters(self):
        """获取模型参数"""
        pass
    
    def set_parameters(self, parameters):
        """设置模型参数"""
        pass
    
    def save(self, path):
        """保存模型"""
        pass
    
    def load(self, path):
        """加载模型"""
        pass
```

## 6. 配置文件设计

### 6.1 YAML 配置结构

```yaml
# 全局配置
global:
  algorithm: PPO  # 强化学习算法
  total_steps: 1000000  # 总训练步数
  log_interval: 1000  # 日志间隔
  save_interval: 10000  # 保存间隔

# Summary 配置
summary:
  enabled: true  # 是否启用 Summary
  type: TensorBoardSummary  # Summary 类型，可选 TensorBoardSummary 或 WandBSummary
  log_dir: ./logs  # TensorBoard 日志目录
  project: x-rl  # Weights & Biases 项目名称
  name: experiment-1  # Weights & Biases 实验名称

# Ray 配置
ray:
  address: "auto"  # Ray 集群地址，"auto" 表示自动检测，分布式部署时设置为 head node 地址
  redis_password: ""  # Redis 密码（如果需要）
  num_cpus: 4  # CPU 数量
  num_gpus: 1  # GPU 数量

# 分布式部署配置
distributed:
  actor_nodes: ["node1", "node2"]  # Actor 运行节点
  learner_nodes: ["node3"]  # Learner 运行节点
  predictor_nodes: ["node3", "node4"]  # Predictor 运行节点（建议 GPU 节点）
  replay_buffer_nodes: ["node5"]  # Replay Buffer 运行节点

# Actor 配置
actor:
  count: 4  # Actor 数量
  batch_size: 64  # 每个 Actor 的批量大小
  max_episode_steps: 1000  # 每个 episode 的最大步数
  nodes:  # Actor 运行节点
    - localhost
    - 192.168.1.100

# Learner 配置
learner:
  batch_size: 256  # 学习批量大小
  learning_rate: 0.0003  # 学习率
  nodes:  # Learner 运行节点
    - localhost

# Predictor 配置
predictor:
  enabled: true  # 是否启用 Predictor
  batch_size: 64  # 批量预测大小
  # 子 Predictor 配置
  sub_predictors:
    predictor_1:
      batch_size: 64
      nodes:
        - localhost
    predictor_2:
      batch_size: 64
      nodes:
        - 192.168.1.101

# Replay Buffer 配置
replay_buffer:
  size: 1000000  # 缓冲区大小
  batch_size: 256  # 采样批量大小
  sampling_strategy: uniform  # 采样策略 (uniform, prioritized)
  nodes:  # Replay Buffer 运行节点
    - localhost

# 环境配置
env:
  name: CartPole-v1  # 环境名称
  kwargs:  # 环境参数
    render_mode: human

# 模型配置
models:
  policy:  # 策略模型
    type: MLP  # 模型类型
    kwargs:  # 模型参数
      hidden_size: 64
      num_layers: 2
  value:  # 价值模型（可选）
    type: MLP  # 模型类型
    kwargs:  # 模型参数
      hidden_size: 64
      num_layers: 2
```

### 6.2 本地训练配置示例

```yaml
# local.yaml
global:
  algorithm: PPO
  total_steps: 1000000
  log_interval: 1000
  save_interval: 10000

actor:
  count: 4
  batch_size: 64
  max_episode_steps: 1000
  nodes:
    - localhost

learner:
  batch_size: 256
  learning_rate: 0.0003
  nodes:
    - localhost

predictor:
  enabled: true
  batch_size: 64
  sub_predictors:
    predictor_1:
      batch_size: 64
      nodes:
        - localhost

replay_buffer:
  size: 1000000
  batch_size: 256
  sampling_strategy: uniform
  nodes:
    - localhost

env:
  name: CartPole-v1
  kwargs:
    render_mode: None

models:
  policy:
    type: MLP
    kwargs:
      hidden_size: 64
      num_layers: 2
  value:
    type: MLP
    kwargs:
      hidden_size: 64
      num_layers: 2
```

### 6.3 分布式训练配置示例

```yaml
# distributed.yaml
global:
  algorithm: PPO
  total_steps: 1000000
  log_interval: 1000
  save_interval: 10000

actor:
  count: 8
  batch_size: 64
  max_episode_steps: 1000
  nodes:
    - 192.168.1.100
    - 192.168.1.101
    - 192.168.1.102
    - 192.168.1.103

learner:
  batch_size: 512
  learning_rate: 0.0003
  nodes:
    - 192.168.1.200

predictor:
  enabled: true
  batch_size: 64
  sub_predictors:
    predictor_1:
      batch_size: 64
      nodes:
        - 192.168.1.100
    predictor_2:
      batch_size: 64
      nodes:
        - 192.168.1.101
    predictor_3:
      batch_size: 64
      nodes:
        - 192.168.1.102
    predictor_4:
      batch_size: 64
      nodes:
        - 192.168.1.103

replay_buffer:
  size: 1000000
  batch_size: 512
  sampling_strategy: uniform
  nodes:
    - 192.168.1.200

env:
  name: CartPole-v1
  kwargs:
    render_mode: None

models:
  policy:
    type: MLP
    kwargs:
      hidden_size: 64
      num_layers: 2
  value:
    type: MLP
    kwargs:
      hidden_size: 64
      num_layers: 2
```

## 7. 运行方式

### 7.1 本地训练

```bash
# 使用本地配置文件运行
python xrl/main.py --config configs/local.yaml
```

### 7.2 分布式训练

1. **启动 Ray 集群**：

```bash
# 在 head 节点上启动
ray start --head --port=6379

# 在 worker 节点上启动
ray start --address=<head-node-ip>:6379
```

2. **运行训练**：

```bash
# 使用分布式配置文件运行
python xrl/main.py --config configs/distributed.yaml
```

### 7.3 main.py 设计

```python
import argparse
import importlib
import ray
from xrl.utils.config import load_config


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="X-RL 训练脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 初始化 Ray
    ray_config = config.get("ray", {})
    ray.init(
        address=ray_config.get("address", "auto"),
        redis_password=ray_config.get("redis_password", ""),
        num_cpus=ray_config.get("num_cpus"),
        num_gpus=ray_config.get("num_gpus")
    )
    
    # 初始化 Controller
    controller_config = config.get("controller", {})
    controller_cls = getattr(
        importlib.import_module("xrl.controllers"),
        controller_config.get("type", "DefaultController")
    )
    controller = controller_cls(config)
    
    # 初始化组件
    controller.initialize()
    
    # 启动训练
    controller.start()
    
    # 监控训练过程
    try:
        controller.monitor()
    except KeyboardInterrupt:
        print("训练被中断")
    finally:
        # 停止训练
        controller.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
```

## 8. 开发流程

1. **定义环境**：实现 `Env` 接口
2. **定义模型**：实现 `Model` 接口
3. **定义智能体**：实现 `Agent` 接口
4. **配置训练参数**：创建 YAML 配置文件
5. **运行训练**：执行主入口文件
6. **监控和调优**：分析训练日志，调整参数

## 9. 示例代码

### 9.1 CartPole 环境实现

```python
# examples/cartpole/env.py
import gymnasium as gym
from xrl.interfaces.env import Env

class CartPoleEnv(Env):
    def __init__(self, render_mode=None):
        self.env = gym.make('CartPole-v1', render_mode=render_mode)
    
    def reset(self):
        state, _ = self.env.reset()
        return state
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done
    
    def close(self):
        self.env.close()
```

### 9.2 CartPole 模型实现

```python
# examples/cartpole/model.py
import torch
import torch.nn as nn
from xrl.interfaces.model import Model

class CartPoleModel(Model):
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.observation_space = 4
        self.action_space = 2
        
        layers = []
        layers.append(nn.Linear(self.observation_space, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, self.action_space))
        self.network = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0003)
    
    def forward(self, state):
        return self.network(torch.FloatTensor(state))
    
    def learn(self, batch):
        # 实现学习逻辑
        # 这里简化处理，实际应该从 batch 中提取数据并计算损失
        loss = torch.tensor(0.0, requires_grad=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def get_parameters(self):
        return self.network.state_dict()
    
    def set_parameters(self, parameters):
        self.network.load_state_dict(parameters)
    
    def save(self, path):
        torch.save(self.network.state_dict(), path)
    
    def load(self, path):
        self.network.load_state_dict(torch.load(path))
```

### 9.3 CartPole 智能体实现

```python
# examples/cartpole/agent.py
import torch
from xrl.interfaces.agent import Agent

class CartPoleAgent(Agent):
    def __init__(self, models, predictors=None):
        super().__init__(models, predictors)
    
    def step(self, obs):
        # 假设我们使用名为 "policy" 的模型进行预测
        model_name = "policy"
        
        if model_name in self.predictors:
            # 使用远程 Predictor 进行预测
            prediction = ray.get(self.predictors[model_name].predict.remote(obs))
        else:
            # 使用本地 Model 进行预测
            prediction = self.models[model_name].forward(obs)
        
        # 根据预测结果生成动作
        action = torch.argmax(torch.tensor(prediction)).item()
        return action
    
    def reset(self):
        # 实现重置逻辑
        pass
```

## 10. 依赖项

```
# requirements.txt
ray
torch
gymnasium
pyyaml
numpy
```

## 11. 安装和使用

1. **安装依赖**：

```bash
pip install -r requirements.txt
```

2. **安装 X-RL**：

```bash
pip install -e .
```

3. **运行示例**：

```bash
python xrl/main.py --config examples/cartpole/config.yaml
```

## 12. 多智能体场景设计

### 12.1 设计目标

支持多智能体场景，如足球比赛、多机器人协作等，其中一个 Actor 可以管理多个 Agent，每个 Agent 控制一个实体（如球员、机器人等）。

### 12.2 实现方案

#### 12.2.1 Actor 设计

```python
@ray.remote
class MultiAgentActor:
    def __init__(self, env, agents, replay_buffer):
        """初始化多智能体 Actor
        
        Args:
            env: 环境实例
            agents: Agent 实例字典，键为 Agent 名称，值为 Agent 实例
            replay_buffer: Replay Buffer 实例
        """
        self.env = env
        self.agents = agents
        self.replay_buffer = replay_buffer
        self.agent_states = {agent_name: None for agent_name in agents}
    
    def run(self):
        """运行多智能体训练"""
        # 重置环境
        states = self.env.reset()
        
        # 重置所有 Agent
        for agent_name, agent in self.agents.items():
            agent.reset()
            self.agent_states[agent_name] = states[agent_name]
        
        while True:
            # 为每个 Agent 生成动作
            actions = {}
            for agent_name, agent in self.agents.items():
                state = self.agent_states[agent_name]
                actions[agent_name] = agent.step(state)
            
            # 执行动作
            next_states, rewards, dones, infos = self.env.step(actions)
            
            # 收集经验数据
            for agent_name, agent in self.agents.items():
                experience = (
                    self.agent_states[agent_name],
                    actions[agent_name],
                    rewards[agent_name],
                    next_states[agent_name],
                    dones[agent_name]
                )
                self.replay_buffer.add.remote(experience)
                self.agent_states[agent_name] = next_states[agent_name]
            
            # 检查是否所有 Agent 都完成
            if all(dones.values()):
                states = self.env.reset()
                for agent_name, agent in self.agents.items():
                    agent.reset()
                    self.agent_states[agent_name] = states[agent_name]
```

#### 12.2.2 多智能体 Agent 设计

以足球比赛为例，每个 Agent 控制一个球员，每个 Agent 包含两个模型：

```python
class FootballPlayerAgent(Agent):
    def __init__(self, models, predictors=None):
        super().__init__(models, predictors)
    
    def step(self, obs):
        # 使用运动模型预测移动方向
        movement_model = "movement"
        if movement_model in self.predictors:
            movement_pred = ray.get(self.predictors[movement_model].predict.remote(obs))
        else:
            movement_pred = self.models[movement_model].forward(obs)
        
        # 使用动作模型预测动作（射门、传球等）
        action_model = "action"
        if action_model in self.predictors:
            action_pred = ray.get(self.predictors[action_model].predict.remote(obs))
        else:
            action_pred = self.models[action_model].forward(obs)
        
        # 组合两个模型的预测结果
        movement = self._process_movement_prediction(movement_pred)
        action = self._process_action_prediction(action_pred)
        
        return {"movement": movement, "action": action}
    
    def _process_movement_prediction(self, prediction):
        # 处理运动预测结果
        pass
    
    def _process_action_prediction(self, prediction):
        # 处理动作预测结果
        pass
    
    def reset(self):
        # 重置 Agent 状态
        pass
```

### 12.3 配置示例

```yaml
# 多智能体配置示例
actor:
  count: 1
  agents:
    player_1:
      type: FootballPlayerAgent
      models:
        movement:
          type: MLP
          kwargs:
            hidden_size: 64
            num_layers: 2
        action:
          type: MLP
          kwargs:
            hidden_size: 64
            num_layers: 2
      predictors:
        movement:
          enabled: true
        action:
          enabled: true
    player_2:
      type: FootballPlayerAgent
      models:
        movement:
          type: MLP
          kwargs:
            hidden_size: 64
            num_layers: 2
        action:
          type: MLP
          kwargs:
            hidden_size: 64
            num_layers: 2
      predictors:
        movement:
          enabled: true
        action:
          enabled: true
```

## 13. 未来规划

- 支持更多强化学习算法
- 提供更丰富的环境接口
- 实现模型自动调优
- 增加可视化工具
- 支持更多分布式训练场景
