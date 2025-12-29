import os
import time
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir="./logs", exp_name=None):
        if exp_name is None:
            exp_name = f"run_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
            
        self.log_dir = os.path.join(log_dir, exp_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"[Logger] Tensorboard logging to {self.log_dir}")

    def log_scalar(self, tag, value, step):
        """记录标量 (Loss, Reward)"""
        self.writer.add_scalar(tag, value, step)

    def log_dict(self, data_dict, step, prefix=""):
        """批量记录字典"""
        for k, v in data_dict.items():
            name = f"{prefix}/{k}" if prefix else k
            self.writer.add_scalar(name, v, step)
            
    def close(self):
        self.writer.close()