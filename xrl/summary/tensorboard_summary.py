"""TensorBoard 实现"""

import os
from datetime import datetime
from xrl.summary.base_summary import BaseSummary


class TensorBoardSummary(BaseSummary):
    """TensorBoard 实现"""
                  
    def __init__(self, config):
        super().__init__(config)
        import tensorflow as tf
        
        base_log_dir = config.get("log_dir", "./logs")
        
        # 生成带时间戳的子目录
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(base_log_dir, timestamp)
        
        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = tf.summary.create_file_writer(log_dir)
        print(f"TensorBoardSummary: 日志目录已创建: {log_dir}")
    
    def scalar(self, tag, value, step):
        with self.writer.as_default():
            import tensorflow as tf
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
    
    def histogram(self, tag, values, step):
        with self.writer.as_default():
            import tensorflow as tf
            tf.summary.histogram(tag, values, step=step)
            self.writer.flush()
    
    def image(self, tag, img_tensor, step):
        with self.writer.as_default():
            import tensorflow as tf
            tf.summary.image(tag, img_tensor, step=step)
            self.writer.flush()
    
    def text(self, tag, text_string, step):
        with self.writer.as_default():
            import tensorflow as tf
            tf.summary.text(tag, text_string, step=step)
            self.writer.flush()
    
    def add_config(self, config):
        with self.writer.as_default():
            import tensorflow as tf
            tf.summary.text("config", str(config), step=0)
            self.writer.flush()
    
    def close(self):
        self.writer.close()
