"""Weights & Biases 实现"""

from xrl.summary.base_summary import BaseSummary


class WandBSummary(BaseSummary):
    """Weights & Biases 实现"""
    
    def __init__(self, config):
        super().__init__(config)
        import wandb
        self.run = wandb.init(
            project=config.get("project", "x-rl"),
            name=config.get("name", None),
            config=config
        )
    
    def scalar(self, tag, value, step):
        self.run.log({tag: value}, step=step)
    
    def histogram(self, tag, values, step):
        import wandb
        self.run.log({tag: wandb.Histogram(values)}, step=step)
    
    def image(self, tag, img_tensor, step):
        import wandb
        self.run.log({tag: wandb.Image(img_tensor)}, step=step)
    
    def text(self, tag, text_string, step):
        self.run.log({tag: text_string}, step=step)
    
    def add_config(self, config):
        self.run.config.update(config)
    
    def close(self):
        self.run.finish()
