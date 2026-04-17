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
        
        # 初始化 Predictors（根据 agents 配置中引用的 predictor 名称）
        predictor_config = self.config.get("predictor", {})
        predictor_cls = getattr(
            importlib.import_module("xrl.core.predictor"),
            predictor_config.get("type", "LocalPredictor")
        )
        predictor_nodes = predictor_config.get("nodes", ["localhost"])
        predictor_node = predictor_nodes[0] if predictor_nodes else "localhost"

        agents_config = self.config.get("agents", {})

        # 收集所有 agent 需要的 predictor 名称
        all_predictor_names = set()
        for agent_config in agents_config.values():
            for name in agent_config.get("predictors", []):
                all_predictor_names.add(name)

        # 为每个 predictor 名称创建一个 Predictor Ray Actor
        self.components["predictors"] = {}
        for predictor_name in all_predictor_names:
            if predictor_name not in self.models:
                print(f"SyncController: 警告 - predictor '{predictor_name}' 对应的模型不存在，跳过")
                continue
            predictor_options = {}
            if predictor_node != "localhost":
                predictor_options["resources"] = {f"node:{predictor_node}": 0.01}
            if predictor_options:
                actor = predictor_cls.options(**predictor_options).remote({predictor_name: self.models[predictor_name]})
            else:
                actor = predictor_cls.remote({predictor_name: self.models[predictor_name]})
            self.components["predictors"][predictor_name] = actor
            print(f"SyncController: Predictor '{predictor_name}' 初始化成功（节点: {predictor_node}）")

        # 初始化 Agents
        self.agents = {}
        for agent_name, agent_config in agents_config.items():
            agent_cls = getattr(
                importlib.import_module(agent_config["module"]),
                agent_config["class"]
            )
            model_names = agent_config.get("models", list(self.models.keys()))
            agent_models = {name: self.models[name] for name in model_names if name in self.models}
            predictor_names = agent_config.get("predictors", [])
            agent_predictors = {name: self.components["predictors"][name]
                                for name in predictor_names
                                if name in self.components["predictors"]}
            self.agents[agent_name] = agent_cls(agent_models, agent_predictors)
        
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
        actor_gamma = actor_config.get("gamma", 0.99)
        actor_lam = actor_config.get("lam", 0.95)

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
                    self.models,
                    actor_gamma,
                    actor_lam
                )
            else:
                actor = actor_cls.remote(
                    env,
                    self.agents,
                    self.components["replay_buffer"],
                    self.models,
                    actor_gamma,
                    actor_lam
                )

            self.components["actors"].append(actor)
            print(f"SyncController: Actor {i+1} 初始化成功（节点: {node}）")
        
        # 初始化 Learner（支持节点分配）
        learner_config = self.config.get("learner", {})
        learner_cls = getattr(
            importlib.import_module("xrl.core.learner"),
            learner_config.get("type", "Learner")
        )

        learner_nodes = learner_config.get("nodes", ["localhost"])
        learner_node = learner_nodes[0] if learner_nodes else "localhost"

        learner_options = {}
        if learner_node != "localhost":
            learner_options["resources"] = {f"node:{learner_node}": 0.01}

        if learner_options:
            self.components["learner"] = learner_cls.options(**learner_options).remote(
                self.models,
                self.components["replay_buffer"],
                self.config
            )
        else:
            self.components["learner"] = learner_cls.remote(
                self.models,
                self.components["replay_buffer"],
                self.config
            )

        print(f"SyncController: Learner 初始化成功（节点: {learner_node}）")

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

                # 调用 Learner 执行训练（Learner 内部负责采样和最小数据量检查）
                print(f"SyncController: 开始训练...")
                success = ray.get(self.components["learner"].train_step.remote())

                if success:
                    self.train_count += 1
                    print(f"SyncController: 训练完成！总训练步数: {self.train_count}")

                    # 从 Learner 获取最新参数并同步给所有 Actors
                    model_params = ray.get(self.components["learner"].get_all_model_parameters.remote())

                    print(f"SyncController: 同步模型参数给所有 Actors...")
                    for actor in self.components["actors"]:
                        actor.update_all_model_parameters.remote(model_params)
                    print(f"SyncController: 所有 Actors 的模型参数已同步")

                    # 同步参数给所有 Predictors
                    for predictor_name, predictor_actor in self.components.get("predictors", {}).items():
                        if predictor_name in model_params:
                            predictor_actor.update_parameters.remote(predictor_name, model_params[predictor_name])
                    if self.components.get("predictors"):
                        print(f"SyncController: 所有 Predictors 的模型参数已同步")
                
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
