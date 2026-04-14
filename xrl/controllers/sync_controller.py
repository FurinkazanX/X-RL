"""同步 Controller 实现 - 保证 Actor 采样参数和 Learner 参数一致（使用 Ray）"""

import importlib
import ray
import time
from typing import Dict, Any
from xrl.controllers.base_controller import BaseController


class SyncController(BaseController):
    """同步 Controller 实现
    
    特点：
    - 使用 Ray Actor 架构
    - 保证 Actor 采样参数和 Learner 参数一致
    - Learner 每次只训练一步
    - 定期同步参数给所有 Actors
    """
    
    def initialize(self):
        """初始化所有组件"""
        print("SyncController: 初始化中...")
        
        # 初始化 Ray
        ray_config = self.config.get("ray", {})
        if not ray.is_initialized():
            ray_address = ray_config.get("address", None)
            if ray_address:
                try:
                    # 尝试连接到现有集群
                    ray.init(
                        address=ray_address,
                        num_cpus=ray_config.get("num_cpus", 4),
                        num_gpus=ray_config.get("num_gpus", 0),
                        ignore_reinit_error=True
                    )
                    print(f"SyncController: Ray 连接到现有集群成功")
                except Exception as e:
                    print(f"SyncController: 连接 Ray 集群失败: {e}")
                    print(f"SyncController: 启动本地 Ray 实例")
                    ray.init(
                        num_cpus=ray_config.get("num_cpus", 2),
                        num_gpus=ray_config.get("num_gpus", 0),
                        ignore_reinit_error=True,
                        include_dashboard=False,
                        _temp_dir="d:/tmp/ray_temp"
                    )
                    print(f"SyncController: Ray 本地实例启动成功")
            else:
                # 直接启动本地 Ray 实例
                print(f"SyncController: 启动本地 Ray 实例")
                ray.init(
                    num_cpus=ray_config.get("num_cpus", 2),
                    num_gpus=ray_config.get("num_gpus", 0),
                    ignore_reinit_error=True,
                    include_dashboard=False
                )
                print(f"SyncController: Ray 本地实例启动成功")
        
        # 初始化 Summary
        summary_config = self.config.get("summary", {})
        if summary_config.get("enabled", False):
            summary_cls = getattr(
                importlib.import_module("xrl.summary"),
                summary_config.get("type", "TensorBoardSummary")
            )
            self.components["summary"] = summary_cls(summary_config)
            self.components["summary"].add_config(self.config)
        
        # 初始化 Models
        models_config = self.config.get("models", {})
        self.models = {}
        for model_name, model_config in models_config.items():
            model_cls = getattr(
                importlib.import_module(model_config["module"]),
                model_config["class"]
            )
            self.models[model_name] = model_cls(**model_config.get("params", {}))
        
        # 初始化 Agents
        agents_config = self.config.get("agents", {})
        self.agents = {}
        for agent_name, agent_config in agents_config.items():
            agent_cls = getattr(
                importlib.import_module(agent_config["module"]),
                agent_config["class"]
            )
            model_names = agent_config.get("models", list(self.models.keys()))
            agent_models = {name: self.models[name] for name in model_names if name in self.models}
            self.agents[agent_name] = agent_cls(agent_models, {})
        
        # 初始化环境
        env_config = self.config.get("env", {})
        env_cls = getattr(
            importlib.import_module(env_config["module"]),
            env_config["class"]
        )
        
        # 初始化 Replay Buffer（支持节点分配）
        replay_buffer_config = self.config.get("replay_buffer", {})
        replay_buffer_cls = getattr(
            importlib.import_module("xrl.core.replay_buffer"),
            replay_buffer_config.get("type", "UniformReplayBuffer")
        )
        
        # 获取 Replay Buffer 节点
        rb_nodes = replay_buffer_config.get("nodes", ["localhost"])
        rb_node = rb_nodes[0] if rb_nodes else "localhost"
        
        # 准备 Replay Buffer options
        rb_options = {}
        if rb_node != "localhost":
            rb_options["resources"] = {f"node:{rb_node}": 0.01}
        
        # 创建 Replay Buffer
        if rb_options:
            self.components["replay_buffer"] = replay_buffer_cls.options(**rb_options).remote(
                replay_buffer_config.get("size", 1000000)
            )
        else:
            self.components["replay_buffer"] = replay_buffer_cls.remote(
                replay_buffer_config.get("size", 1000000)
            )
        
        print(f"SyncController: Replay Buffer 初始化成功，大小: {replay_buffer_config.get('size', 1000000)}（节点: {rb_node}）")
        
        # 初始化 Actors（使用 Ray Actor，支持节点分配）
        actor_config = self.config.get("actor", {})
        actor_count = actor_config.get("count", 1)
        actor_cls = getattr(
            importlib.import_module("xrl.core.actor"),
            actor_config.get("type", "Actor")
        )
        
        # 获取 Actor 节点列表
        actor_nodes = actor_config.get("nodes", ["localhost"])
        
        self.components["actors"] = []
        for i in range(actor_count):
            # 每个 Actor 创建独立的环境实例
            env = env_cls(**env_config.get("params", {}))
            
            # 选择节点（轮询分配）
            node = actor_nodes[i % len(actor_nodes)]
            
            # 准备 Actor options
            actor_options = {}
            if node != "localhost":
                # 指定节点资源
                actor_options["resources"] = {f"node:{node}": 0.01}
            
            # 创建 Actor
            if actor_options:
                actor = actor_cls.options(**actor_options).remote(
                    env,
                    self.agents,
                    self.components["replay_buffer"],
                    self.models
                )
            else:
                actor = actor_cls.remote(
                    env,
                    self.agents,
                    self.components["replay_buffer"],
                    self.models
                )
            
            self.components["actors"].append(actor)
            print(f"SyncController: Actor {i+1} 初始化成功（节点: {node}）")
        
        # 初始化 Learner 配置
        learner_config = self.config.get("learner", {})
        self.epochs = learner_config.get("epochs", 4)
        self.batch_size = learner_config.get("batch_size", 256)
        print(f"SyncController: Learner 配置 - epochs: {self.epochs}, batch_size: {self.batch_size}")
        
        print(f"SyncController: 所有组件初始化完成！")
    
    def start(self):
        """启动训练 - 同步模式（使用 Ray）"""
        self.running = True
        print("=" * 80)
        print("SyncController: 开始同步训练（使用 Ray）")
        print("=" * 80)
        
        try:
            while self.running:
                # 让所有 Actor 运行一个 episode 并收集数据
                episode_futures = []
                for actor in self.components["actors"]:
                    future = actor.run_episode.remote(write_to_buffer=False)
                    episode_futures.append(future)
                
                # 等待所有 Actor 完成
                results = ray.get(episode_futures)
                
                # 处理数据并写入 Replay Buffer
                all_experiences = []
                for result in results:
                    episode_info = result["episode_info"]
                    all_processed_experiences = result["all_processed_experiences"]
                    
                    self.episode_count += 1
                    episode_reward = sum(episode_info["episode_rewards"].values())
                    print(f"SyncController: Episode {self.episode_count} 完成 - reward: {episode_reward:.2f}, length: {episode_info['episode_length']}")
                    
                    if all_processed_experiences:
                        all_experiences.extend(all_processed_experiences)
                
                # 写入 Replay Buffer
                if all_experiences:
                    for exp in all_experiences:
                        self.components["replay_buffer"].add.remote(exp)
                    print(f"SyncController: 共 {len(all_experiences)} 条经验写入 Replay Buffer")
                
                # 检查是否有足够数据训练
                buffer_size = ray.get(self.components["replay_buffer"].get_size.remote())
                print(f"SyncController: Replay Buffer 大小: {buffer_size}")
                
                if buffer_size >= self.batch_size:
                    experiences = ray.get(self.components["replay_buffer"].sample.remote(self.batch_size))
                    
                    if experiences and len(experiences) > 0:
                        print(f"SyncController: 开始训练...")
                        for model_name, model in self.models.items():
                            model.learn(experiences)
                        
                        self.train_count += 1
                        print(f"SyncController: 训练完成！总训练步数: {self.train_count}")
                        
                        # 同步参数给所有 Actors
                        print(f"SyncController: 同步模型参数给所有 Actors...")
                        model_params = {}
                        for model_name, model in self.models.items():
                            if hasattr(model, 'get_parameters'):
                                model_params[model_name] = model.get_parameters()
                        
                        for actor in self.components["actors"]:
                            actor.update_all_model_parameters.remote(model_params)
                        
                        print(f"SyncController: 所有 Actors 的模型参数已同步")
                
                print(f"\n{'=' * 60}\n")
                
                # 记录到 Summary
                if "summary" in self.components:
                    self.components["summary"].scalar("episode/reward", episode_reward, self.episode_count)
                    self.components["summary"].scalar("episode/length", episode_info["episode_length"], self.episode_count)
                    self.components["summary"].scalar("train/count", self.train_count, self.episode_count)
        
        except KeyboardInterrupt:
            print(f"\nSyncController: 收到停止信号，正在停止...")
        
        self.stop()
    
    def stop(self):
        """停止所有组件"""
        self.running = False
        
        # 关闭 Summary
        if "summary" in self.components:
            self.components["summary"].close()
        
        # 关闭 Ray
        if ray.is_initialized():
            ray.shutdown()
        
        print("=" * 80)
        print("SyncController: 训练已停止")
        print("=" * 80)
