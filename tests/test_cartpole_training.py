"""测试 CartPole 模型训练 - 使用修复后的 learn 方法"""

import numpy as np
import torch
import gymnasium as gym
from examples.cartpole.model import CartPoleModel
from examples.cartpole.agent import CartPoleAgent
from xrl.types import PPOBatch


def collect_trajectory(env, agent, max_steps=500):
    """收集一条轨迹"""
    state, _ = env.reset()
    episode_reward = 0
    step_infos = []
    
    for step in range(max_steps):
        # 选择动作
        action, step_info = agent.step(state)
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 更新 step_info
        step_info.update(reward, done, next_state, action)
        step_infos.append(step_info)
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    return step_infos, episode_reward


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """计算 GAE 优势"""
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


def train():
    """训练模型"""
    # 创建环境
    env = gym.make("CartPole-v1")
    
    # 创建模型
    model = CartPoleModel(input_dim=4, action_dim=2, hidden_dims=[64, 64], lr=3e-4)
    
    # 创建智能体
    agent = CartPoleAgent({"main": model}, {})
    
    # 训练参数
    num_episodes = 500
    update_interval = 20
    
    rewards_history = []
    all_trajectories = []
    
    for episode in range(num_episodes):
        # 收集轨迹
        trajectory, episode_reward = collect_trajectory(env, agent)
        all_trajectories.append(trajectory)
        rewards_history.append(episode_reward)
        
        # 每10个episode打印一次平均奖励
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}, Average Reward (last 10): {avg_reward:.2f}")
        
        # 定期更新模型
        if (episode + 1) % update_interval == 0 and len(all_trajectories) > 0:
            # 准备训练数据
            all_states, all_actions, all_rewards = [], [], []
            all_next_states, all_dones, all_log_probs, all_values, all_advantages = [], [], [], [], []
            
            for traj in all_trajectories:
                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []
                log_probs = []
                values = []
                
                for i, step_info in enumerate(traj):
                    states.append(step_info.state)
                    actions.append(step_info.action)
                    rewards.append(step_info.reward)
                    dones.append(step_info.done)
                    
                    # 获取 next_state
                    if i < len(traj) - 1:
                        next_states.append(traj[i + 1].state)
                    else:
                        next_states.append(step_info.next_state)
                    
                    # 获取 log_prob 和 value
                    if hasattr(step_info, 'log_prob') and step_info.log_prob is not None:
                        log_probs.append(step_info.log_prob)
                    else:
                        log_probs.append(0.0)
                    
                    if hasattr(step_info, 'value') and step_info.value is not None:
                        values.append(step_info.value)
                    else:
                        values.append(0.0)
                
                # 计算优势
                advantages = compute_gae(rewards, values, dones)
                
                all_states.extend(states)
                all_actions.extend(actions)
                all_rewards.extend(rewards)
                all_next_states.extend(next_states)
                all_dones.extend(dones)
                all_log_probs.extend(log_probs)
                all_values.extend(values)
                all_advantages.extend(advantages)
            
            # 创建 PPOBatch
            batch = PPOBatch(
                states=all_states,
                actions=all_actions,
                rewards=all_rewards,
                next_states=all_next_states,
                dones=all_dones,
                log_probs=all_log_probs,
                values=all_values,
                advantages=all_advantages
            )
            
            # 训练模型（多个 epoch）
            for _ in range(4):
                model.learn(batch)
            
            # 清空轨迹
            all_trajectories = []
    
    env.close()
    
    # 打印最终平均奖励
    final_avg = np.mean(rewards_history[-100:])
    print(f"\nFinal average reward (last 100 episodes): {final_avg:.2f}")
    
    # 绘制奖励曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('CartPole Training Progress')
    plt.savefig('cartpole_training_progress.png')
    print("Training progress saved to cartpole_training_progress.png")


if __name__ == "__main__":
    train()
