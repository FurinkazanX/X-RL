"""Predictor 模块初始化"""

from xrl.core.predictor.base_predictor import BasePredictor
from xrl.core.predictor.local_predictor import LocalPredictor
from xrl.core.predictor.distributed_predictor import DistributedPredictor

__all__ = ["BasePredictor", "LocalPredictor", "DistributedPredictor"]
