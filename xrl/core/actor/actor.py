"""Actor 实现"""

import ray
from typing import Dict, Any, List
from xrl.core.actor.base_actor import BaseActor
from xrl.core.types import Experience, Trajectory


@ray.remote
class Actor(BaseActor):
    """Actor 实现"""
    
    def run_episode(self, write_to_buffer: bool = True) -> Dict[str, Any]:
        """执行一个 episode，收集完整轨迹，计算 GAE
        
        Args:
            write_to_buffer: 是否写入 Replay Buffer
        
        Returns:
            包含 episode_info 和 all_processed_experiences 的字典
        """
        # 重置环境
        state = self.env.reset()
        # 重置所有智能体
        for agent_id, agent in self.agents.items():
            agent.reset()
        done = {agent_id: False for agent_id in self.agents}
        episode_rewards = {agent_id: 0.0 for agent_id in self.agents}
        episode_length = 0
        
        # 收集完整轨迹，按 model_name 分组
        step_infos_by_model = {}  # {model_name: [step_infos]}
        
        while not all(done.values()):
            # 每个智能体选择动作
            actions = {}
            step_infos = {}
            for agent_id, agent in self.agents.items():
                if not done[agent_id]:
                    action, step_info = agent.step(state)
                    actions[agent_id] = action
                    step_infos[agent_id] = step_info
            
            # 执行动作
            next_state, reward, done, info = self.env.step(actions)
            
            # 更新每个智能体的 step_info 对象
            for agent_id, step_info in step_infos.items():
                step_info.update(reward[agent_id], done[agent_id], next_state, actions[agent_id], info)
                
                # 按 model_name 分组
                model_name = step_info.model_name
                if model_name not in step_infos_by_model:
                    step_infos_by_model[model_name] = []
                step_infos_by_model[model_name].append(step_info)
                
                # 累加奖励
                episode_rewards[agent_id] += reward[agent_id]
            
            episode_length += 1
            # 更新状态
            state = next_state
        
        # 处理完整轨迹（计算 GAE 和 returns）
        all_processed_experiences = []
        if step_infos_by_model and self.models:
            try:
                for model_name, step_infos in step_infos_by_model.items():
                    if model_name not in self.models:
                        print(f"Actor: 跳过 model {model_name}，未在 models 中找到")
                        continue
                    
                    trajectory = Trajectory(step_infos)
                    model = self.models[model_name]
                    processed_experiences = model.process_trajectory(trajectory, gamma=self.gamma, lam=self.lam)
                    
                    all_processed_experiences.extend(processed_experiences)
                
            except Exception as e:
                print(f"Actor: 处理轨迹失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 写入 Replay Buffer（如果需要）
        if write_to_buffer and all_processed_experiences:
            try:
                for exp in all_processed_experiences:
                    self.replay_buffer.add.remote(exp)
                
                print(f"Actor: Episode 轨迹处理完成，写入 {len(all_processed_experiences)} 条经验")
            
            except Exception as e:
                print(f"Actor: 写入 Replay Buffer 失败: {e}")
                import traceback
                traceback.print_exc()
        
        episode_info = {
            "episode_rewards": episode_rewards,
            "episode_length": episode_length
        }
        
        return {
            "episode_info": episode_info,
            "all_processed_experiences": all_processed_experiences
        }
    
    def run(self) -> None:
        """运行 Actor，与环境交互并收集经验"""
        while True:
            result = self.run_episode()
            episode_info = result["episode_info"]
            episode_reward = sum(episode_info["episode_rewards"].values())
            print(f"Actor: Episode finished, reward: {episode_reward:.2f}, length: {episode_info['episode_length']}")
    
    def update_all_model_parameters(self, model_params: Dict[str, Dict[str, Any]]) -> None:
        """更新所有模型的参数

        Args:
            model_params: 模型参数字典 {model_name: parameters}
        """
        print(f"Actor: 收到参数同步请求，准备更新 {len(model_params)} 个模型")

        for model_name, params in model_params.items():
            if model_name in self.models and hasattr(self.models[model_name], 'set_parameters'):
                self.models[model_name].set_parameters(params)
                print(f"Actor: 模型 {model_name} 参数已更新")
    
    def reset(self) -> None:
        """重置 Actor 状态"""
        self.env.reset()
