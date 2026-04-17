"""异步 Controller 实现 - 不保证 Actor 和 Learner 参数一致"""

import importlib
import ray
import time
from typing import Dict, Any
from xrl.controllers.base_controller import BaseController


class AsyncController(BaseController):
    """异步 Controller 实现
    
    特点：
    - Actor 自由将数据存入 Replay Buffer
    - Learner 也自由从 Replay Buffer 获取数据
    - 无需保证 Actor 和 Learner 的神经网络参数一致
    - 使用 Ray Actor 实现分布式部署
    """
    
    def initialize(self):
        """初始化所有 Ray 组件"""
        # 初始化 Ray
        ray_config = self.config.get("ray", {})
        if not ray.is_initialized():
            try:
                ray.init(
                    address=ray_config.get("address", "auto"),
                    num_cpus=ray_config.get("num_cpus", 4),
                    num_gpus=ray_config.get("num_gpus", 0),
                    ignore_reinit_error=True
                )
                print(f"AsyncController: Ray 连接到现有集群成功")
            except Exception:
                print(f"AsyncController: 未找到现有 Ray 集群，启动本地实例")
                ray.init(
                    num_cpus=ray_config.get("num_cpus", 2),
                    num_gpus=ray_config.get("num_gpus", 0),
                    ignore_reinit_error=True,
                    include_dashboard=False,
                    _temp_dir="d:/tmp/ray_temp"
                )
                print(f"AsyncController: Ray 本地实例启动成功")
        
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
        
        # 初始化环境（保存配置供每个 Actor 独立创建实例）
        env_config = self.config.get("env", {})
        env_cls = getattr(
            importlib.import_module(env_config["module"]),
            env_config["class"]
        )

        # 初始化 Replay Buffer
        replay_buffer_config = self.config.get("replay_buffer", {})
        replay_buffer_cls = getattr(
            importlib.import_module("xrl.core.replay_buffer"),
            replay_buffer_config.get("type", "UniformReplayBuffer")
        )
        self.components["replay_buffer"] = replay_buffer_cls.remote(
            replay_buffer_config.get("size", 1000000)
        )
        print(f"AsyncController: Replay Buffer 初始化成功，大小: {replay_buffer_config.get('size', 1000000)}")

        # 初始化 Actors（支持多个）
        actor_config = self.config.get("actor", {})
        actor_count = actor_config.get("count", 1)
        actor_cls = getattr(
            importlib.import_module("xrl.core.actor"),
            actor_config.get("type", "Actor")
        )
        actor_gamma = actor_config.get("gamma", 0.99)
        actor_lam = actor_config.get("lam", 0.95)

        self.components["actors"] = []
        for i in range(actor_count):
            # 每个 Actor 创建独立的环境实例
            env = env_cls(**env_config.get("params", {}))
            actor = actor_cls.remote(
                env,
                self.agents,
                self.components["replay_buffer"],
                self.models,
                actor_gamma,
                actor_lam
            )
            self.components["actors"].append(actor)
            print(f"AsyncController: Actor {i+1} 初始化成功")
        
        # 初始化 Learner
        learner_config = self.config.get("learner", {})
        learner_cls = getattr(
            importlib.import_module("xrl.core.learner"),
            learner_config.get("type", "Learner")
        )
        self.components["learner"] = learner_cls.remote(
            self.models,
            self.components["replay_buffer"],
            self.config
        )
        print(f"AsyncController: Learner 初始化成功")
        
        # 初始化 Predictor（如果启用）
        predictor_config = self.config.get("predictor", {})
        if predictor_config.get("enabled", False):
            predictor_cls = getattr(
                importlib.import_module("xrl.core.predictor"),
                predictor_config.get("type", "LocalPredictor")
            )
            self.components["predictor"] = predictor_cls.remote(self.models)
            print(f"AsyncController: Predictor 初始化成功")
        
        print(f"AsyncController: 所有组件初始化完成！")
    
    def start(self):
        """启动训练 - 异步模式"""
        self.running = True
        print("=" * 80)
        print("AsyncController: 开始异步训练")
        print("=" * 80)
        
        # 启动 Actors
        if "actors" in self.components and self.components["actors"]:
            for i, actor in enumerate(self.components["actors"]):
                actor.run.remote()
                print(f"AsyncController: Actor {i+1} 已启动")
        
        # 启动 Learner
        if "learner" in self.components:
            self.components["learner"].train.remote()
            print(f"AsyncController: Learner 已启动")
        
        print(f"AsyncController: 训练进行中...（按 Ctrl+C 停止）")
        
        try:
            while self.running:
                time.sleep(1.0)
                # 检查 Replay Buffer 大小
                if "replay_buffer" in self.components:
                    try:
                        buffer_size = ray.get(self.components["replay_buffer"].get_size.remote(), timeout=1.0)
                        print(f"AsyncController: Replay Buffer 大小: {buffer_size}")
                    except Exception as e:
                        print(f"AsyncController: 获取 Replay Buffer 大小失败: {e}")
                
        except KeyboardInterrupt:
            print(f"\nAsyncController: 收到停止信号，正在停止...")
        
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
        print("AsyncController: 训练已停止")
        print("=" * 80)
