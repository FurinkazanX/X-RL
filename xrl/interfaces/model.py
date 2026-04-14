"""Model 接口定义"""

from typing import Dict, Any, Optional
from typing import Dict, Any, List
from xrl.core.types import Trajectory, Experience


class Model:
    """模型接口"""
    
    def __init__(self, **kwargs):
        """初始化模型
        
        Args:
            **kwargs: 模型参数
        """
        pass
    
    def forward(self, inputs: Dict[str, Any]) -> Any:
        """前向传播，返回动作和 StepInfo 对象
        
        Args:
            inputs: 输入数据
        
        Returns:
            包含动作和 StepInfo 对象的元组
        """
        raise NotImplementedError
    
    def learn(self, batch: Any) -> None:
        """更新模型参数
        
        Args:
            batch: 批量数据，具体类型由算法决定
        """
        raise NotImplementedError
    
    @classmethod
    def process_trajectory(cls, trajectory: Trajectory, **kwargs) -> List[Experience]:
        """处理完整轨迹，返回处理后的经验数据列表
        
        Args:
            trajectory: 完整的轨迹数据
            **kwargs: 其他参数
        
        Returns:
            处理后的经验数据列表
        """
        # 默认实现：直接返回原始经验数据
        return trajectory.experiences
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数
        
        Returns:
            模型参数
        """
        raise NotImplementedError
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """设置模型参数
        
        Args:
            parameters: 模型参数
        """
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """保存模型
        
        Args:
            path: 保存路径
        """
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """加载模型
        
        Args:
            path: 加载路径
        """
        raise NotImplementedError
