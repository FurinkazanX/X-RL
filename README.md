# X-RL: 可扩展的强化学习框架

X-RL 是一个模块化、可扩展的强化学习框架，支持多种算法（PPO、DQN、SAC），易于使用和二次开发。

## 特性

- **多算法支持**: 内置 PPO（近端策略优化）、DQN（深度 Q 网络）、SAC（软 Actor-Critic）
- **模块化设计**: 清晰的模块划分（Actor、Learner、Predictor、Replay Buffer）
- **灵活的配置**: 基于 YAML 的配置系统
- **分布式训练**: 支持 Ray 分布式训练
- **开箱即用**: 提供 CartPole 和 Pendulum 等经典环境的完整示例

## 项目结构

```
X-RL/
├── xrl/                      # 核心框架代码
│   ├── algo/                 # 算法实现
│   │   ├── ppo/             # PPO 算法
│   │   ├── dqn/             # DQN 算法
│   │   └── sac/             # SAC 算法
│   ├── controllers/          # 控制器
│   ├── core/                 # 核心组件
│   │   ├── actor/            # Actor（采样器）
│   │   ├── learner/          # Learner（学习器）
│   │   ├── predictor/        # Predictor（预测器）
│   │   └── replay_buffer/    # 经验回放池
│   ├── interfaces/           # 接口定义
│   ├── summary/              # 摘要记录（TensorBoard、WandB）
│   └── utils/                # 工具函数
├── examples/                 # 示例环境
│   ├── cartpole/             # CartPole 示例
│   └── pendulum/             # Pendulum 示例
└── run.py                    # 训练入口脚本
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

#### CartPole (离散动作空间)

```bash
python run.py --config examples/cartpole/config.yaml
```

#### Pendulum (连续动作空间)

```bash
python run.py --config examples/pendulum/config.yaml
```

## 核心概念

### 1. 接口

框架定义了三个核心接口：

- **Env**: 环境接口，定义与环境交互的方法
- **Agent**: 智能体接口，定义决策逻辑
- **Model**: 模型接口，定义网络结构和学习逻辑

### 2. 组件

- **Actor**: 负责与环境交互，收集轨迹数据
- **Learner**: 负责从经验中学习，更新模型参数
- **Predictor**: 负责提供模型预测服务
- **Replay Buffer**: 存储经验数据，支持均匀采样和优先经验回放
- **Controller**: 协调整个训练流程

### 3. PPO 算法特点

- **连续和离散动作空间**: 统一支持
- **广义优势估计 (GAE)**: 高效的优势函数计算
- **策略裁剪**: 防止策略更新过大
- **熵正则化**: 鼓励探索
- **批量计算优化**: log\_prob 在学习阶段批量计算，提升效率

## 如何自定义

### 添加新算法

1. 在 `xrl/algo/` 下创建新目录
2. 实现模型类（继承 `PPOModel` 或 `Model`）
3. 实现对应的 StepInfo 类

### 添加新环境

1. 在 `examples/` 下创建新目录
2. 实现环境类（继承 `Env`）
3. 实现智能体类（继承 `Agent`）
4. 实现模型类（继承 `PPOModel` 等）
5. 创建配置文件 `config.yaml`

### 配置说明

配置文件主要包含以下部分：

```yaml
env:           # 环境配置
agent:         # 智能体配置
model:         # 模型配置
actor:         # Actor 配置
learner:       # Learner 配置
predictor:     # Predictor 配置
replay_buffer: # 经验回放池配置
controller:    # 控制器配置
summary:       # 摘要记录配置
```

## 技术亮点

### 1. 数值稳定性

- **标准差限制**: 连续动作空间中限制标准差范围 \[0.1, 1.0]，防止 NaN
- **梯度裁剪**: 限制梯度范数，防止梯度爆炸
- **优势标准化**: 稳定训练过程

### 2. 代码结构

- **单一职责**: 每个方法只做一件事
- **清晰的命名**: 方法和变量名一目了然
- **模块化**: 易于测试和维护

### 3. 性能优化

- **批量计算**: log\_prob 在学习阶段批量计算，而非逐个样本计算
- **高效的经验回放**: 支持多种采样策略

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！
