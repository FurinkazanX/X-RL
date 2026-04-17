"""默认 Controller 实现 - 使用 Ray Actor 架构"""

import importlib
import ray
import time
from typing import Dict, Any
from xrl.controllers.base_controller import BaseController


class DefaultController(BaseController):
    """默认 Controller 实现 - 使用 Ray Actor 架构
    
    组件包括：
    - Replay Buffer (Ray Actor): 存储经验数据
    - Actor (Ray Actor): 与环境交互收集经验（支持多个）
    - Learner (Ray Actor): 从 Replay Buffer 采样数据进行训练
    - Predictor (Ray Actor，可选): 批量预测
    """
    
    def initialize(self):
        """初始化所有 Ray 组件"""
        # 初始化 Ray
        ray_config = self.config.get("ray", {})
        if not ray.is_initialized():
            try:
                # 尝试连接到现有集群
                ray.init(
                    address=ray_config.get("address", "auto"),
                    num_cpus=ray_config.get("num_cpus", 4),
                    num_gpus=ray_config.get("num_gpus", 0),
                    ignore_reinit_error=True
                )
                print(f"Controller: Ray 连接到现有集群成功")
            except Exception:
                # 如果无法连接到现有集群，启动本地 Ray 实例
                print(f"Controller: 未找到现有 Ray 集群，启动本地实例")
                # 简化本地启动配置，避免资源问题
                ray.init(
                    num_cpus=ray_config.get("num_cpus", 2),  # 减少 CPU 数量
                    num_gpus=ray_config.get("num_gpus", 0),
                    ignore_reinit_error=True,
                    include_dashboard=False,  # 禁用 dashboard，减少启动开销
                    _temp_dir="d:/tmp/ray_temp"  # 使用临时目录
                )
                print(f"Controller: Ray 本地实例启动成功 (CPUs: {ray_config.get('num_cpus', 2)}, GPUs: {ray_config.get('num_gpus', 0)})")
        
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
                print(f"Controller: 警告 - predictor '{predictor_name}' 对应的模型不存在，跳过")
                continue
            predictor_options = {}
            if predictor_node != "localhost":
                predictor_options["resources"] = {f"node:{predictor_node}": 0.01}
            if predictor_options:
                actor = predictor_cls.options(**predictor_options).remote({predictor_name: self.models[predictor_name]})
            else:
                actor = predictor_cls.remote({predictor_name: self.models[predictor_name]})
            self.components["predictors"][predictor_name] = actor
            print(f"Controller: Predictor '{predictor_name}' 初始化成功（节点: {predictor_node}）")

        # 初始化 Agents（传入对应的 predictors）
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
        
        # 初始化环境（保存配置供每个 Actor 独立创建实例）
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
        
        print(f"Controller: Replay Buffer 初始化成功，大小: {replay_buffer_config.get('size', 1000000)}（节点: {rb_node}）")
        
        # 初始化 Actors（支持多个和节点分配）
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
            print(f"Controller: Actor {i+1} 初始化成功（节点: {node}）")
        
        # 初始化 Learner（支持节点分配）
        learner_config = self.config.get("learner", {})
        learner_cls = getattr(
            importlib.import_module("xrl.core.learner"),
            learner_config.get("type", "Learner")
        )
        
        # 获取 Learner 节点
        learner_nodes = learner_config.get("nodes", ["localhost"])
        learner_node = learner_nodes[0] if learner_nodes else "localhost"
        
        # 准备 Learner options
        learner_options = {}
        if learner_node != "localhost":
            learner_options["resources"] = {f"node:{learner_node}": 0.01}
        
        # 创建 Learner
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
        
        print(f"Controller: Learner 初始化成功（节点: {learner_node}）")
        
        print(f"Controller: 所有组件初始化完成！")
    
    def start(self):
        """启动训练 - 使用 Ray Actor 架构"""
        self.running = True
        print("=" * 80)
        print("Controller: 开始分布式训练")
        print("=" * 80)
        
        # 从配置读取参数同步间隔
        controller_config = self.config.get("controller", {})
        sync_step_interval = controller_config.get("sync_step_interval", 10)  # 默认每 10 个 train step 同步一次
        print(f"Controller: 模型参数同步间隔设置为每 {sync_step_interval} 个 train step")
        
        # 启动 Actors
        if "actors" in self.components and self.components["actors"]:
            for i, actor in enumerate(self.components["actors"]):
                actor.run.remote()
                print(f"Controller: Actor {i+1} 已启动")
        
        # 启动 Learner（async 方法）
        if "learner" in self.components:
            self.components["learner"].train.remote()
            print(f"Controller: Learner 已启动")
        
        print(f"Controller: 训练进行中...（按 Ctrl+C 停止）")
        
        # 监控训练状态，根据 train step 同步参数
        last_synced_step = 0
        
        try:
            while self.running:
                time.sleep(1.0)
                # 检查 Replay Buffer 大小
                if "replay_buffer" in self.components:
                    try:
                        buffer_size = ray.get(self.components["replay_buffer"].get_size.remote(), timeout=1.0)
                        print(f"Controller: Replay Buffer 大小: {buffer_size}")
                    except Exception as e:
                        print(f"Controller: 获取 Replay Buffer 大小失败: {e}")
                
                # 检查当前训练步数，决定是否同步参数
                if "learner" in self.components:
                    try:
                        # 现在 Learner 是 async 的，查询不会被阻塞了
                        current_step = ray.get(self.components["learner"].get_train_step_count.remote(), timeout=2.0)
                        if current_step - last_synced_step >= sync_step_interval and current_step > 0:
                            print(f"Controller: 当前训练步数 {current_step}，开始同步模型参数...")
                            self._sync_model_parameters()
                            last_synced_step = current_step
                    
                    except Exception as e:
                        print(f"Controller: 获取训练步数失败: {e}")
                
        except KeyboardInterrupt:
            print(f"\nController: 收到停止信号，正在停止...")
        
        self.stop()
    
    def _sync_model_parameters(self):
        """同步模型参数：从 Learner 获取，然后更新到所有 Actors"""
        if "learner" not in self.components or "actors" not in self.components:
            return
        
        try:
            print("=" * 60)
            print("Controller: 开始同步模型参数...")
            
            # 从 Learner 获取最新参数（现在 Learner 是 async 的，查询不会被阻塞了）
            model_params = ray.get(self.components["learner"].get_all_model_parameters.remote(), timeout=5.0)
            
            print(f"Controller: 从 Learner 获取到 {len(model_params)} 个模型的参数")
            
            if model_params:
                # 更新所有 Actors 的模型参数（Actor 内部会同步传播到 Predictor）
                for i, actor in enumerate(self.components["actors"]):
                    actor.update_all_model_parameters.remote(model_params)
                    print(f"Controller: 已发送参数更新请求给 Actor {i+1}")
                print(f"Controller: 模型参数已同步，更新了 {len(self.components['actors'])} 个 Actors")
            
            print("=" * 60)
        
        except Exception as e:
            print(f"Controller: 同步模型参数失败: {e}")
            import traceback
            traceback.print_exc()
    
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
        print("Controller: 训练已停止")
        print("=" * 80)
    
    def monitor(self):
        """监控系统运行状态"""
        # 监控逻辑
        pass
