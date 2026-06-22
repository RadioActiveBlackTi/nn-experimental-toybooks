import torch

class Interpolator:
    # Variance-Preserving Interpolation
    def interpolate(self, x0, x1, z, t):
        return x1 * t + z * torch.sqrt(1 - t**2)
    
    def dt_interpolate(self, x0, x1, z, t):
        return x1 - (t / torch.sqrt(1 - t**2)) * z
    
    def interpolate_both(self, x0, x1, z, t):
        interp = self.interpolate(x0, x1, z, t)
        dt_interp = self.dt_interpolate(x0, x1, z, t)
        return interp, dt_interp

class OTInterpolator:
    # Optimal Transport (Rectified Flow) Interpolation
    def interpolate(self, x0, x1, z, t):
        return (1 - t) * x0 + t * x1
    
    def dt_interpolate(self, x0, x1, z, t):
        return x1 - x0
    
    def interpolate_both(self, x0, x1, z, t):
        return self.interpolate(x0, x1, z, t), self.dt_interpolate(x0, x1, z, t)