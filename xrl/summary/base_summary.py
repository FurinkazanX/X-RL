"""Summary 基类接口"""

from typing import Dict, Any, Optional


class BaseSummary:
    """Summary 基类"""
    
    def __init__(self, config):
        """初始化 Summary
        
        Args:
            config: 配置对象
        """
        self.config = config
    
    def scalar(self, tag, value, step):
        """记录标量值
        
        Args:
            tag: 标签名称
            value: 标量值
            step: 训练步数
        """
        pass
    
    def histogram(self, tag, values, step):
        """记录直方图
        
        Args:
            tag: 标签名称
            values: 数据值
            step: 训练步数
        """
        pass
    
    def image(self, tag, img_tensor, step):
        """记录图像
        
        Args:
            tag: 标签名称
            img_tensor: 图像张量
            step: 训练步数
        """
        pass
    
    def text(self, tag, text_string, step):
        """记录文本
        
        Args:
            tag: 标签名称
            text_string: 文本内容
            step: 训练步数
        """
        pass
    
    def add_config(self, config):
        """添加配置信息
        
        Args:
            config: 配置对象
        """
        pass
    
    def close(self):
        """关闭 Summary"""
        pass
