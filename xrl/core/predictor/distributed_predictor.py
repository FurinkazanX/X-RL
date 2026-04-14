"""分布式 Predictor 实现"""

import ray
import time
import threading
from typing import Dict, Any, List, Optional
from xrl.core.predictor.base_predictor import BasePredictor


@ray.remote
class DistributedPredictor(BasePredictor):
    """分布式 Predictor 实现"""
    
    def __init__(self, models: Dict[str, Any]):
        super().__init__(models)
        self.batch_size = 64  # 批量大小
        self.timeout = 0.1  # 超时时间（秒）
        self.requests = []  # 预测请求队列
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self._process_batch, daemon=True)
        self.thread.start()
    
    def predict(self, model_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """预测接口
        
        Args:
            model_name: 模型名称
            state: 状态数据
        
        Returns:
            预测结果
        """
        # 创建一个 Future 对象来接收结果
        future = ray.util.ActorPool.make_future()
        
        with self.condition:
            self.requests.append((model_name, state, future))
            # 如果达到 batch_size，唤醒处理线程
            if len(self.requests) >= self.batch_size:
                self.condition.notify()
        
        # 等待结果
        return future.get()
    
    def _process_batch(self) -> None:
        """处理批量预测请求"""
        while True:
            with self.condition:
                # 等待直到有足够的请求或超时
                ready = self.condition.wait(timeout=self.timeout)
                # 即使超时，也处理当前积累的请求
                if len(self.requests) > 0:
                    batch = self.requests.copy()
                    self.requests.clear()
                else:
                    continue
            
            # 按模型分组请求
            model_requests = {}
            for model_name, state, future in batch:
                if model_name not in model_requests:
                    model_requests[model_name] = []
                model_requests[model_name].append((state, future))
            
            # 对每个模型执行批量预测
            for model_name, requests in model_requests.items():
                if model_name not in self.models:
                    for _, future in requests:
                        future.set_exception(ValueError(f"Model {model_name} not found"))
                    continue
                
                # 准备批量输入
                states = [req[0] for req in requests]
                batch_input = {"state": states}
                
                # 执行批量预测
                try:
                    results = self.models[model_name].forward(batch_input)
                    
                    # 分发结果
                    for (_, future), result in zip(requests, results):
                        # 检查结果类型
                        if isinstance(result, tuple) and len(result) == 2:
                            # 模型返回的是 (action, step_info) 元组
                            action, step_info = result
                            # 提取模型输出信息
                            model_output = step_info.model_output if hasattr(step_info, 'model_output') else {}
                            # 构造返回字典
                            future.set_result({
                                "action": action,
                                "step_info": step_info,
                                **model_output
                            })
                        else:
                            # 保持向后兼容，处理返回字典的情况
                            future.set_result(result)
                except Exception as e:
                    for _, future in requests:
                        future.set_exception(e)
    
    def update_parameters(self, model_name: str, parameters: Dict[str, Any]) -> None:
        """更新模型参数
        
        Args:
            model_name: 模型名称
            parameters: 模型参数
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        self.models[model_name].set_parameters(parameters)
