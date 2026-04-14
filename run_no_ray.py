"""不使用 Ray 的完整训练 - 使用所有框架代码"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import importlib
import time
import numpy as np
from xrl.utils.config import load_config

from examples.cartpole.env import CartPoleEnv
from examples.cartpole.agent import CartPoleAgent
from examples.cartpole.model import CartPoleModel
from xrl.algo.ppo.model import PPOModel
from xrl.core.types import Trajectory


def main():
    print("=" * 80)
    print("X-RL - 不使用 Ray 的完整训练")
    print("=" * 80)
    
    # 加载配置
    config = load_config("examples/cartpole/config.yaml")
    
    print("\n初始化组件...")
    
    # 1. 初始化环境
    env_config = config.get("env", {})
    env_cls = getattr(
        importlib.import_module(env_config["module"]),
        env_config["class"]
    )
    env = env_cls(**env_config.get("params", {}))
    
    # 2. 初始化模型
    models_config = config.get("models", {})
    models = {}
    for model_name, model_config in models_config.items():
        print(f"  加载模型: {model_name}")
        model_cls = getattr(
            importlib.import_module(model_config["module"]),
            model_config["class"]
        )
        models[model_name] = model_cls(**model_config.get("params", {}))
    
    # 3. 初始化 Agents
    agents_config = config.get("agents", {})
    agents = {}
    for agent_name, agent_config in agents_config.items():
        print(f"  加载 Agent: {agent_name}")
        agent_cls = getattr(
            importlib.import_module(agent_config["module"]),
            agent_config["class"]
        )
        model_names = agent_config.get("models", list(models.keys()))
        agent_models = {name: models[name] for name in model_names if name in models}
        agents[agent_name] = agent_cls(agent_models, {})
    
    print("\n初始化完成！开始训练...")
    print("=" * 80)
    print()
    
    num_episodes = 500
    update_interval = 10
    all_experiences = []
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = {agent_id: False for agent_id in agents}
        trajectory = {agent_id: [] for agent_id in agents}
        
        # 运行一个 episode
        while not all(done.values()):
            actions = {}
            step_infos = {}
            
            for agent_id, agent in agents.items():
                if not done[agent_id]:
                    action, step_info = agent.step(state)
                    actions[agent_id] = action
                    step_infos[agent_id] = step_info
            
            next_state, reward, done, info = env.step(actions)
            
            for agent_id, step_info in step_infos.items():
                step_info.update(reward[agent_id], done[agent_id], next_state, actions[agent_id], info)
                trajectory[agent_id].append(step_info)
            
            state = next_state
        
        # 处理轨迹
        for agent_id, step_infos in trajectory.items():
            if step_infos:
                traj = Trajectory(step_infos)
                processed = PPOModel.process_trajectory(traj)
                all_experiences.extend(processed)
                
                # 记录奖励
                last_step = step_infos[-1]
                if hasattr(last_step, 'info') and 'episode_reward' in last_step.info:
                    rewards_history.append(last_step.info['episode_reward'])
                else:
                    rewards_history.append(sum(exp.reward for exp in processed))
        
        if (episode + 1) % 10 == 0:
            avg = np.mean(rewards_history[-10:]) if rewards_history else 0
            curr = rewards_history[-1] if rewards_history else 0
            print(f"Episode {episode+1:3d} | Last 10 avg: {avg:6.1f} | Current: {curr:4.0f}")
        
        if (episode + 1) % update_interval == 0 and all_experiences:
            for model_name, model in models.items():
                model.learn(all_experiences)
            all_experiences = []
    
    print()
    print("=" * 80)
    print("训练完成！")
    
    final_avg = np.mean(rewards_history[-20:]) if len(rewards_history) >= 20 else 0
    max_reward = max(rewards_history) if rewards_history else 0
    print(f"最终 20 个 episode 平均奖励: {final_avg:.1f}")
    print(f"最高奖励: {max_reward}")
    print("=" * 80)
    
    env.close()


if __name__ == "__main__":
    main()
