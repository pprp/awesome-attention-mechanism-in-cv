from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random
import time
import os


class TensorboardLogger():
    def __init__(self, log_dir="tfboard_logs"):
        self.writer = SummaryWriter()

    def add_scalar(self, title, name, scalar_value, global_step):
        self.writer.add_scalar(title+'/'+name, scalar_value, global_step)
