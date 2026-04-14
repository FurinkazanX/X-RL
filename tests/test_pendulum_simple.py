"""简单的 Pendulum 测试脚本 - 独立于框架，快速验证"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class SimplePendulumModel(nn.Module):
    """简单的 Pendulum 模型"""
    
    def __init__(self, input_dim=3, action_dim=1, hidden_dims=[64, 64], lr=3e-4):
        super().__init__()
        
        # 策略网络
        policy_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            policy_layers.append(nn.Linear(prev_dim, hidden_dim))
            policy_layers.append(nn.Tanh())
            prev_dim = hidden_dim
        self.policy_net = nn.Sequential(*policy_layers)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Parameter(torch.zeros(action_dim))
        
        # 价值网络
        value_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            value_layers.append(nn.Linear(prev_dim, hidden_dim))
            value_layers.append(nn.Tanh())
            prev_dim = hidden_dim
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_net = nn.Sequential(*value_layers)
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + 
            list(self.mean_head.parameters()) + 
            [self.log_std_head] + 
            list(self.value_net.parameters()),
            lr=lr
        )
    
    def get_action(self, state):
        """获取动作"""
        x = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            policy_hidden = self.policy_net(x)
            mean = self.mean_head(policy_hidden)
            std = torch.exp(self.log_std_head)
            
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -2.0, 2.0)
            
            log_prob = dist.log_prob(action).sum(dim=-1).item()
            value = self.value_net(x).squeeze(-1).item()
        
        return action.squeeze(0).numpy(), log_prob, value
    
    def learn(self, experiences):
        """学习"""
        states = torch.FloatTensor([exp["state"] for exp in experiences])
        actions = torch.FloatTensor([exp["action"] for exp in experiences])
        old_log_probs = torch.FloatTensor([exp["log_prob"] for exp in experiences])
        advantages = torch.FloatTensor([exp["advantage"] for exp in experiences])
        returns = torch.FloatTensor([exp["return"] for exp in experiences])
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 训练多个 epoch
        for epoch in range(10):
            # 前向传播
            policy_hidden = self.policy_net(states)
            mean = self.mean_head(policy_hidden)
            std = torch.exp(self.log_std_head)
            values_pred = self.value_net(states).squeeze(-1)
            
            # 计算新的 log_prob
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions.unsqueeze(1)).sum(dim=-1)
            
            # PPO 损失
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = nn.MSELoss()(values_pred, returns)
            
            total_loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """计算 GAE"""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def main():
    """主函数"""
    print("开始 Pendulum 简单测试")
    
    env = gym.make("Pendulum-v1")
    model = SimplePendulumModel(lr=3e-4)
    
    num_episodes = 200
    update_interval = 10
    all_experiences = []
    rewards_history = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_experiences = []
        
        while not done:
            action, log_prob, value = model.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            
            episode_experiences.append({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "log_prob": log_prob,
                "value": value
            })
            
            state = next_state
        
        # 处理轨迹
        rewards = [exp["reward"] for exp in episode_experiences]
        values = [exp["value"] for exp in episode_experiences]
        dones = [exp["done"] for exp in episode_experiences]
        
        advantages, returns = compute_gae(rewards, values, dones)
        
        for i, exp in enumerate(episode_experiences):
            exp["advantage"] = advantages[i]
            exp["return"] = returns[i]
            all_experiences.append(exp)
        
        rewards_history.append(episode_reward)
        
        # 更新
        if (episode + 1) % update_interval == 0 and all_experiences:
            model.learn(all_experiences)
            all_experiences = []
        
        if (episode + 1) % 10 == 0:
            avg = np.mean(rewards_history[-10:]) if rewards_history else 0
            print(f"Episode {episode+1:3d} | Last 10 avg: {avg:7.1f} | Current: {episode_reward:7.1f}")
    
    env.close()
    print("\n测试完成！")


if __name__ == "__main__":
    main()
