"""PPO 模型基类"""

from typing import List, Dict, Union
from xrl.interfaces.model import Model
from xrl.core.types import Trajectory, Experience
import numpy as np
import torch
import torch.nn.functional as F


class PPOModel(Model):
    """PPO 模型基类
    
    支持单动作头和多动作头：
    - 单动作头：action 是 tensor，dist_params 是 {'action': {...}}
    - 多动作头：action 是 dict，dist_params 是 {'head1': {...}, 'head2': {...}}
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clip_epsilon = kwargs.get('clip_epsilon', 0.2)
        self.value_coef = kwargs.get('value_coef', 0.5)
        self.entropy_coef = kwargs.get('entropy_coef', 0.01)
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        self.optimizer = None
        self._policy_params = []
        self._value_params = []
    
    def learn(self, experiences: List[Experience]) -> None:
        if not experiences:
            return
        
        states, actions, advantages, returns, old_log_probs = self._prepare_data(experiences)
        self._train_epochs(states, actions, advantages, returns, old_log_probs)
    
    def _extract_actions(self, experiences: List[Experience]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """提取动作，支持单动作头和多动作头"""
        actions_list = [exp.action for exp in experiences]
        
        if isinstance(actions_list[0], dict):
            result = {}
            for key in actions_list[0].keys():
                key_actions = np.array([a[key] for a in actions_list])
                result[key] = torch.LongTensor(key_actions) if np.issubdtype(key_actions.dtype, np.integer) else torch.FloatTensor(key_actions)
            return result
        else:
            actions = np.array(actions_list)
            return torch.LongTensor(actions) if np.issubdtype(actions.dtype, np.integer) else torch.FloatTensor(actions)
    
    def _compute_log_probs(self, dist_params_list: List[Dict], actions: Union[torch.Tensor, Dict]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """从 dist_params 列表计算 log_probs
        
        Args:
            dist_params_list: dist_params 列表
                - 计算 old_log_probs 时：每个元素是单个 experience 的 dist_params（logits 是 numpy）
                - 计算 new_log_probs 时：只有一个元素，是模型输出的 dist_params（logits 是带梯度的 Tensor）
            actions: 动作（tensor 或 dict）
            
        Returns:
            log_probs: tensor 或 dict of tensors
        """
        first_dp = dist_params_list[0]
        is_single_action = 'action' in first_dp and len(first_dp) == 1
        
        result = {}
        for head_name in first_dp.keys():
            head_actions = actions if is_single_action else actions[head_name]
            head_dps = [dp[head_name] for dp in dist_params_list]
            
            first_head_dp = head_dps[0]
            dist_type = first_head_dp.get('type')
            
            if dist_type == 'continuous':
                result[head_name] = self._compute_continuous_log_probs(head_dps, head_actions)
            elif dist_type == 'categorical':
                result[head_name] = self._compute_categorical_log_probs(head_dps, head_actions)
            else:
                raise ValueError(f"Unknown dist_type: {dist_type}")
        
        return result['action'] if is_single_action else result
    
    def _compute_continuous_log_probs(self, dist_params_list: List[Dict], actions: torch.Tensor) -> torch.Tensor:
        """计算连续动作空间的 log_probs"""
        from xrl.algo.ppo.distribution import Continuous
        
        means, stds = [], []
        for dp in dist_params_list:
            means.append(dp['mean'])
            stds.append(dp['std'])
        
        # 判断是否为带梯度的 Tensor（来自模型前向传播）
        is_tensor = isinstance(means[0], torch.Tensor) and means[0].requires_grad
        
        if is_tensor:
            # 来自模型输出，保留梯度流
            if len(means) == 1:
                means = means[0]
                stds = stds[0]
            else:
                means = torch.stack(means)
                stds = torch.stack(stds)
        else:
            # 来自 experiences（numpy 或无梯度 Tensor），转为 FloatTensor
            means = [m.detach().numpy() if isinstance(m, torch.Tensor) else m for m in means]
            stds = [s.detach().numpy() if isinstance(s, torch.Tensor) else s for s in stds]
            means = torch.FloatTensor(np.array(means))
            stds = torch.FloatTensor(np.array(stds))
        
        if means.dim() == 3 and means.shape[1] == 1:
            means = means.squeeze(1)
        if stds.dim() == 3 and stds.shape[1] == 1:
            stds = stds.squeeze(1)
        if means.dim() == 1:
            means = means.unsqueeze(1)
        if stds.dim() == 1:
            stds = stds.unsqueeze(1)
        
        dist = Continuous(means, stds)
        batch_actions = actions.unsqueeze(1) if actions.dim() == 1 and means.dim() > 1 else actions
        return dist.log_prob(batch_actions)
    
    def _compute_categorical_log_probs(self, dist_params_list: List[Dict], actions: torch.Tensor) -> torch.Tensor:
        """计算离散动作空间的 log_probs"""
        from xrl.algo.ppo.distribution import Categorical
        
        all_logits = []
        for dp in dist_params_list:
            all_logits.append(dp['logits'])
        
        # 判断是否为带梯度的 Tensor（来自模型前向传播）
        is_tensor = isinstance(all_logits[0], torch.Tensor) and all_logits[0].requires_grad
        
        if is_tensor:
            # 来自模型输出，保留梯度流
            if len(all_logits) == 1:
                logits = all_logits[0]
            else:
                logits = torch.stack(all_logits)
        else:
            # 来自 experiences（numpy 或无梯度 Tensor），转为 FloatTensor
            all_logits = [l.detach().numpy() if isinstance(l, torch.Tensor) else l for l in all_logits]
            logits = torch.FloatTensor(np.array(all_logits))
        
        dist = Categorical(logits)
        
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        
        return dist.log_prob(actions)
    
    def _compute_entropy(self, dist_params: Dict[str, Dict], actions: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """从 dist_params 计算 entropy（保留梯度流，用于熵奖励）"""
        from xrl.algo.ppo.distribution import Categorical, Continuous
        
        is_single_action = 'action' in dist_params and len(dist_params) == 1
        
        total_entropy = 0.0
        for head_name, params in dist_params.items():
            dist_type = params.get('type')
            
            if dist_type == "categorical":
                dist = Categorical(params["logits"])
            elif dist_type == "continuous":
                mean = params["mean"]
                std = params["std"]
                dist = Continuous(mean, std)
            else:
                raise ValueError(f"Unknown dist_type: {dist_type}")
            
            total_entropy += dist.entropy().mean()
        
        return total_entropy
    
    def _prepare_data(self, experiences: List[Experience]):
        """从 experiences 中提取并准备数据"""
        states = [exp.state for exp in experiences]
        actions = self._extract_actions(experiences)
        advantages = [exp.advantage for exp in experiences]
        returns = [exp.return_ for exp in experiences]
        
        states = torch.FloatTensor(np.array(states))
        
        dist_params_list = []
        for exp in experiences:
            dp = None
            if hasattr(exp, 'info') and 'model_output' in exp.info and 'dist_params' in exp.info['model_output']:
                dp = exp.info['model_output']['dist_params']
            dist_params_list.append(dp)
        
        if any(dp is None for dp in dist_params_list):
            raise ValueError("部分经验缺少 dist_params，无法计算 old_log_probs")
        
        old_log_probs = self._compute_log_probs(dist_params_list, actions)
        
        advantages = torch.FloatTensor(np.array(advantages))
        returns = torch.FloatTensor(np.array(returns))
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, advantages, returns, old_log_probs
    
    def _train_epochs(self, states, actions, advantages, returns, old_log_probs):
        """训练多个 epoch"""
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        batch_size = 64
        
        for epoch in range(4):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = self._index_data(actions, batch_indices)
                batch_old_log_probs = self._index_data(old_log_probs, batch_indices)
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                self._train_step(batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns)
    
    def _index_data(self, data: Union[torch.Tensor, Dict], indices: np.ndarray) -> Union[torch.Tensor, Dict]:
        """索引 tensor 或 dict of tensors"""
        if isinstance(data, dict):
            return {key: act[indices] for key, act in data.items()}
        return data[indices]
    
    def _train_step(self, batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns):
        """单个训练步骤"""
        model_output = self.forward({"state": batch_states}, train=True)
        values_pred = model_output["value"]
        
        new_log_probs = self._compute_log_probs([model_output["dist_params"]], batch_actions)
        entropy = self._compute_entropy(model_output["dist_params"], batch_actions)
        
        total_loss = self._compute_loss(new_log_probs, batch_old_log_probs, batch_advantages, values_pred, batch_returns, entropy)
        
        self._optimize(total_loss)
    
    def _compute_loss(self, new_log_probs, old_log_probs, advantages, values_pred, returns, entropy):
        """计算 PPO 损失，支持多动作头"""
        if isinstance(new_log_probs, dict):
            total_policy_loss = 0.0
            for head_name in new_log_probs.keys():
                ratio = torch.exp(new_log_probs[head_name] - old_log_probs[head_name])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                total_policy_loss += -torch.min(surr1, surr2).mean()
        else:
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            total_policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = F.mse_loss(values_pred, returns)
        return total_policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
    
    def _optimize(self, total_loss):
        """执行优化步骤
        
        如果子类设置了 _policy_params 和 _value_params，则分别裁剪梯度，
        避免价值网络的大梯度淹没策略网络的梯度。
        否则，统一裁剪所有参数的梯度。
        """
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            total_loss.backward()
            
            if self._policy_params and self._value_params:
                policy_params = [p for p in self._policy_params if p.grad is not None]
                value_params = [p for p in self._value_params if p.grad is not None]
                if policy_params:
                    torch.nn.utils.clip_grad_norm_(policy_params, max_norm=self.max_grad_norm)
                if value_params:
                    torch.nn.utils.clip_grad_norm_(value_params, max_norm=self.max_grad_norm)
            else:
                params = []
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            params.append(p)
                torch.nn.utils.clip_grad_norm_(params, max_norm=self.max_grad_norm)
            
            self.optimizer.step()
    
    @classmethod
    def process_trajectory(cls, trajectory: Trajectory, **kwargs) -> List[Experience]:
        """处理完整轨迹，计算 GAE 和 returns"""
        gamma = kwargs.get('gamma', 0.99)
        lam = kwargs.get('lam', 0.95)
        value_estimator = kwargs.get('value_estimator', None)
        
        rewards = [exp.reward for exp in trajectory.experiences]
        dones = [exp.done for exp in trajectory.experiences]
        
        if value_estimator:
            values = [value_estimator.estimate(exp.state) for exp in trajectory.experiences]
        else:
            values = []
            for exp in trajectory.experiences:
                if hasattr(exp, 'value') and exp.value is not None:
                    values.append(exp.value)
                elif "value" in exp.info.get("model_output", {}):
                    values.append(exp.info["model_output"]["value"])
                else:
                    values.append(0.0)
        
        advantages = cls._compute_gae(rewards, values, dones, gamma, lam)
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        processed_experiences = []
        for i, exp in enumerate(trajectory.experiences):
            exp.advantage = advantages[i]
            exp.return_ = returns[i]
            processed_experiences.append(exp)
        
        return processed_experiences
    
    @staticmethod
    def _compute_gae(rewards: List[float], values: List[float], dones: List[bool], gamma: float, lam: float) -> List[float]:
        """计算 GAE 优势"""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            next_value = 0.0 if t == n - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages.tolist()
