import numpy as np
import torch

class SineWaveTask:
    def __init__(self):
        self.amplitude = np.random.uniform(0.1, 5.0)
        self.phase = np.random.uniform(0, np.pi)
        
    def sample_data(self, size=10):
        x = np.random.uniform(-5.0, 5.0, size)
        y = self.amplitude * np.sin(x + self.phase)
        
        return torch.tensor(x, dtype=torch.float32).unsqueeze(1), \
               torch.tensor(y, dtype=torch.float32).unsqueeze(1)