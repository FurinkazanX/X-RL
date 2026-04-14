"""Pendulum 模型实现"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from xrl.algo.ppo.ppo_model import PPOModel
from xrl.algo.ppo.distribution import Continuous


class PendulumModel(PPOModel):
    """Pendulum 完整模型，包含策略网络和价值网络"""
    
    def __init__(self, input_dim=3, action_dim=1, hidden_dims=[256, 256], lr=3e-4, **kwargs):
        """初始化模型
        
        Args:
            input_dim: 输入维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            lr: 学习率
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # 策略网络：输出均值和对数标准差
        policy_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            policy_layers.append(nn.Linear(prev_dim, hidden_dim))
            policy_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.policy_net = nn.Sequential(*policy_layers)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Parameter(torch.ones(action_dim) * -0.5)
        
        # 价值网络
        value_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            value_layers.append(nn.Linear(prev_dim, hidden_dim))
            value_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_net = nn.Sequential(*value_layers)
        
        self._policy_params = (
            list(self.policy_net.parameters()) + 
            list(self.mean_head.parameters()) + 
            [self.log_std_head]
        )
        self._value_params = list(self.value_net.parameters())
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self._policy_params + self._value_params,
            lr=lr
        )
    
    def forward(self, inputs: dict, train: bool = False) -> dict:
        """前向传播，统一接口
        
        Args:
            inputs: 输入数据，包含 state 等字段
            train: 是否为训练模式
        
        Returns:
            train=True: 包含 dist_params 和 value 的字典
            train=False: 包含 action 和 step_info 的字典
        """
        x = inputs["state"]
        
        # 将输入转换为 torch tensor
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        if train:
            policy_hidden = self.policy_net(x)
            mean = self.mean_head(policy_hidden)
            mean = torch.clamp(mean, min=-2.0, max=2.0)
            
            std = torch.exp(self.log_std_head)
            std = torch.clamp(std, min=1e-3, max=1.0)
            
            value = self.value_net(x).squeeze(-1)
            
            return {
                "dist_params": {
                    "action": {
                        "type": "continuous", 
                        "mean": mean, 
                        "std": std
                    }
                },
                "value": value
            }
        else:
            with torch.no_grad():
                policy_hidden = self.policy_net(x)
                mean = self.mean_head(policy_hidden)
                mean = torch.clamp(mean, min=-2.0, max=2.0)
                
                std = torch.exp(self.log_std_head)
                std = torch.clamp(std, min=1e-3, max=1.0)
                
                value = self.value_net(x).squeeze(-1).item()
                
                dist = Continuous(mean, std)
                action = dist.sample()
                action = torch.clamp(action, min=-2.0, max=2.0)
            
            action_np = action.squeeze(0).numpy()
            actions = {"action": action_np}
            
            model_output = {
                "actions": actions,
                "value": value,
                "dist_params": {
                    "action": {
                        "type": "continuous",
                        "mean": mean.squeeze(0).numpy(),
                        "std": std.squeeze(0).numpy()
                    }
                }
            }
            
            from xrl.algo.ppo.ppo_step_info import PPOStepInfo
            step_info = PPOStepInfo(state=inputs["state"], model_output=model_output, 
                                    model_name=inputs.get("model_name", "pendulum_model"))
            step_info.value = value
            
            return {
                "actions": actions,
                "step_info": step_info
            }

    
    def get_parameters(self) -> dict:
        """获取模型参数
        
        Returns:
            模型参数
        """
        import numpy as np
        
        def tensor_to_numpy(tensor_or_dict):
            if isinstance(tensor_or_dict, torch.Tensor):
                return tensor_or_dict.cpu().numpy()
            elif isinstance(tensor_or_dict, dict):
                return {k: tensor_to_numpy(v) for k, v in tensor_or_dict.items()}
            return tensor_or_dict
        
        return {
            "policy_net": tensor_to_numpy(self.policy_net.state_dict()),
            "mean_head": tensor_to_numpy(self.mean_head.state_dict()),
            "log_std_head": self.log_std_head.data.cpu().numpy(),
            "value_net": tensor_to_numpy(self.value_net.state_dict())
        }
    
    def set_parameters(self, parameters) -> None:
        """设置模型参数
        
        Args:
            parameters: 模型参数
        """
        import numpy as np
        
        def numpy_to_tensor(numpy_or_dict):
            if isinstance(numpy_or_dict, np.ndarray):
                return torch.tensor(numpy_or_dict)
            elif isinstance(numpy_or_dict, dict):
                return {k: numpy_to_tensor(v) for k, v in numpy_or_dict.items()}
            return numpy_or_dict
        
        if "policy_net" in parameters:
            self.policy_net.load_state_dict(numpy_to_tensor(parameters["policy_net"]))
        if "mean_head" in parameters:
            self.mean_head.load_state_dict(numpy_to_tensor(parameters["mean_head"]))
        if "log_std_head" in parameters:
            self.log_std_head.data = numpy_to_tensor(parameters["log_std_head"])
        if "value_net" in parameters:
            self.value_net.load_state_dict(numpy_to_tensor(parameters["value_net"]))
