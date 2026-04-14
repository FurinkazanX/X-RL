"""测试 PPO 训练脚本"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import List, Tuple

class SimplePPOModel(nn.Module):
    """简单的 PPO 模型"""
    
    def __init__(self, input_dim=4, action_dim=2, hidden_dim=64):
        super().__init__()
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        """前向传播"""
        logits = self.policy_net(state)
        value = self.value_net(state)
        return logits, value
    
    def get_action(self, state):
        """获取动作"""
        with torch.no_grad():
            logits, value = self.forward(state)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs[0, action[0]] + 1e-8)
            return action.item(), log_prob.item(), value.item()


def collect_trajectory(env, model, max_steps=500) -> Tuple[List, float]:
    """收集一条轨迹"""
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
    
    state, _ = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        # 转换为 tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 获取动作
        action, log_prob, value = model.get_action(state_tensor)
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 存储经验
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        dones.append(done)
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'log_probs': log_probs,
        'values': values,
        'dones': dones
    }, episode_reward


def compute_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    """计算优势函数"""
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
    
    return advantages


def train_ppo():
    """训练 PPO"""
    # 创建环境
    env = gym.make("CartPole-v1")
    
    # 创建模型
    model = SimplePPOModel(input_dim=4, action_dim=2, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # 训练参数
    num_episodes = 500
    update_interval = 20  # 每20个episode更新一次
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    
    rewards_history = []
    all_trajectories = []
    
    for episode in range(num_episodes):
        # 收集轨迹
        trajectory, episode_reward = collect_trajectory(env, model)
        all_trajectories.append(trajectory)
        rewards_history.append(episode_reward)
        
        # 每10个episode打印一次平均奖励
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}, Average Reward (last 10): {avg_reward:.2f}")
        
        # 定期更新模型
        if (episode + 1) % update_interval == 0 and len(all_trajectories) > 0:
            # 准备训练数据
            all_states, all_actions, all_old_log_probs, all_advantages, all_returns = [], [], [], [], []
            
            for traj in all_trajectories:
                states = torch.FloatTensor(traj['states'])
                actions = torch.LongTensor(traj['actions'])
                old_log_probs = torch.FloatTensor(traj['log_probs'])
                rewards = traj['rewards']
                values = traj['values']
                dones = traj['dones']
                
                # 计算优势
                advantages = compute_advantages(rewards, values, dones)
                advantages = torch.FloatTensor(advantages)
                
                # 计算回报
                returns = [adv + val for adv, val in zip(advantages, values)]
                returns = torch.FloatTensor(returns)
                
                all_states.append(states)
                all_actions.append(actions)
                all_old_log_probs.append(old_log_probs)
                all_advantages.append(advantages)
                all_returns.append(returns)
            
            # 合并所有数据
            states = torch.cat(all_states, dim=0)
            actions = torch.cat(all_actions, dim=0)
            old_log_probs = torch.cat(all_old_log_probs, dim=0)
            advantages = torch.cat(all_advantages, dim=0)
            returns = torch.cat(all_returns, dim=0)
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 训练多个 epoch
            for _ in range(4):  # 4个epoch
                # 前向传播
                logits, values = model(states)
                
                # 计算新的动作概率
                probs = torch.softmax(logits, dim=-1)
                new_log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
                
                # 计算比率
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # PPO 损失
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = nn.MSELoss()(values.squeeze(-1), returns)
                
                # 熵损失
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                
                # 总损失
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 清空轨迹
            all_trajectories = []
    
    env.close()
    
    # 绘制奖励曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('PPO Training Progress')
    plt.savefig('ppo_training_progress.png')
    print("Training progress saved to ppo_training_progress.png")
    
    # 打印最终平均奖励
    final_avg = np.mean(rewards_history[-100:])
    print(f"\nFinal average reward (last 100 episodes): {final_avg:.2f}")

if __name__ == "__main__":
    train_ppo()
