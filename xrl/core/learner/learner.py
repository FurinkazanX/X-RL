"""通用 Learner 实现"""

import ray
import time
import asyncio
import psutil
from typing import Dict, Any, Optional
from xrl.core.learner.base_learner import BaseLearner


@ray.remote
class Learner(BaseLearner):
    """通用 Learner 实现
    
    不局限于某个特定算法，而是调用模型的 learn 方法来执行具体的强化学习算法训练
    """
    
    def __init__(self, models: Dict[str, Any], replay_buffer: Any, config: Optional[Dict[str, Any]] = None):
        super().__init__(models, replay_buffer, config)
        self.last_train_time = time.time()
        self.train_step_count = 0  # 训练步数计数
        self.running = False  # 控制训练循环的标志
    
    def train_step(self) -> bool:
        """执行单次训练步骤
        
        Returns:
            是否成功执行训练
        """
        # 检查是否应该训练
        if not self.should_train():
            return False
        
        # 调整批量大小
        batch_size = self.adjust_batch_size()
        
        # 检查 Replay Buffer 大小
        try:
            buffer_size = ray.get(self.replay_buffer.get_size.remote())
        except Exception as e:
            return False
        
        # 最小数据量要求
        min_buffer_size = batch_size * 2
        
        if buffer_size < min_buffer_size:
            return False
        
        # 从 Replay Buffer 采样数据
        try:
            experiences = ray.get(self.replay_buffer.sample.remote(batch_size), timeout=5.0)
        except Exception as e:
            print(f"采样数据失败: {e}")
            return False
        
        if not experiences or len(experiences) < 10:
            return False
        
        print(f"Learner: 采样到 {len(experiences)} 条经验，开始训练")
        
        for model_name, model in self.models.items():
            model.learn(experiences)
        
        print(f"Learner: 训练完成！当前训练步数: {self.train_step_count + 1}")
        
        # 更新最后训练时间和步数计数
        self.last_train_time = time.time()
        self.train_step_count += 1
        return True
    
    async def train(self) -> None:
        """训练模型（async 版本）
        
        通用训练逻辑：从 Replay Buffer 采样数据，然后调用每个模型的 learn 方法进行训练
        """
        self.running = True
        print(f"Learner: 开始 async 训练循环")
        
        while self.running:
            # 使用线程池运行同步的 train_step，避免阻塞事件循环
            success = await asyncio.to_thread(self.train_step)
            # 如果训练失败（数据不足），等待一下再重试
            if not success:
                await asyncio.sleep(0.1)
    
    def stop_training(self) -> None:
        """停止训练循环"""
        self.running = False
        print(f"Learner: 停止训练循环")
    
    def get_train_step_count(self) -> int:
        """获取当前训练步数
        
        Returns:
            训练步数
        """
        return self.train_step_count
    
    def get_all_model_parameters(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型的参数（用于同步给 Actor）
        
        Returns:
            所有模型的参数字典 {model_name: parameters}
        """
        all_params = {}
        for model_name, model in self.models.items():
            if hasattr(model, 'get_parameters'):
                all_params[model_name] = model.get_parameters()
        return all_params
    
    def update_parameters(self) -> None:
        """更新模型参数并同步给 Predictor"""
        # 这里可以实现参数同步逻辑
        pass
