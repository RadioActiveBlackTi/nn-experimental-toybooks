import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from schedulers import schedule_sigma, schedule_sigma_linear, schedule_sigma_karras, scheduler_sigma_polynomial

class NCSN_sampler:
    # Langevin Dynamics Sampler
    def __init__(self, score_model, scheduler=schedule_sigma(), device='cpu'):
        self.score_model = score_model
        self.scheduler = scheduler
        self.device = device
    
    def sample(self, batch_size, num_batches, num_steps=1000, step_lr=2e-5):
        self.score_model.eval()
        result = np.zeros((num_batches * batch_size, 2), dtype=np.float32)

        t_final = torch.ones(1, device=self.device)
        sigma_min = self.scheduler.get_sigma(t_final).item()

        with torch.no_grad():
            for i in range(num_batches):
                t_init = torch.zeros(batch_size, device=self.device)
                sigma_max = self.scheduler.get_sigma(t_init).view(-1, 1)
                x = torch.randn(batch_size, 2).to(self.device) * sigma_max

                for step in range(num_steps):
                    t = torch.full((batch_size,), step / num_steps, device=self.device)
                    sigma_curr = self.scheduler.get_sigma(t).view(-1, 1)
                    
                    score = self.score_model(x, sigma_curr) 
                    
                    alpha = step_lr * (sigma_curr / sigma_min) ** 2
                    
                    z = torch.randn_like(x)
                    
                    if step == num_steps - 1:
                        z = torch.zeros_like(x)

                    x = x + 0.5 * alpha * score + torch.sqrt(alpha) * z
                    
                result[i * batch_size:(i + 1) * batch_size] = x.detach().cpu().numpy()
        return result
    
    def sample_with_history(self, batch_size, num_steps=1000, delim=10, step_lr=2e-5):
        self.score_model.eval()
        
        t_init = torch.zeros(batch_size, device=self.device)
        sigma_max = self.scheduler.get_sigma(t_init).view(-1, 1)
        t_final = torch.ones(1, device=self.device)
        sigma_min = self.scheduler.get_sigma(t_final).item()
        x = torch.randn(batch_size, 2).to(self.device) * sigma_max
        history = [x.detach().cpu().numpy()]

        with torch.no_grad():
            for step in range(num_steps):
                t = torch.full((batch_size,), step / num_steps, device=self.device)
                sigma_curr = self.scheduler.get_sigma(t).view(-1, 1)
                
                score = self.score_model(x, sigma_curr) 
                
                alpha = step_lr * (sigma_curr / sigma_min) ** 2
                
                z = torch.randn_like(x)
                
                if step == num_steps - 1:
                    z = torch.zeros_like(x)

                x = x + 0.5 * alpha * score + torch.sqrt(alpha) * z
                if step % delim == 0:
                    history.append(x.detach().cpu().numpy())
        return np.array(history)

class DDPM_sampler:
    # DDPM style Sampler
    # But for VE-SDE
    def __init__(self, score_model, scheduler=schedule_sigma(), device='cpu'):
        self.score_model = score_model
        self.scheduler = scheduler
        self.device = device
    
    def sample(self, batch_size, num_batches, num_steps=1000):
        self.score_model.eval()
        
        result = np.zeros((num_batches * batch_size, 2), dtype=np.float32)

        with torch.no_grad():
            t_init = torch.zeros(batch_size, device=self.device)
            sigma_max = self.scheduler.get_sigma(t_init).view(-1, 1)
            x = torch.randn(batch_size, 2).to(self.device) * sigma_max
            for i in range(num_batches):
                x = torch.randn(batch_size, 2).to(self.device)
                for step in range(num_steps):
                    t = torch.full((batch_size,), step / num_steps, device=self.device)
                    sigma_curr = self.scheduler.get_sigma(t).view(-1, 1)
                    sigma_next = self.scheduler.get_sigma(torch.clamp(t + 1.0 / num_steps, max=1.0)).view(-1, 1)

                    score = self.score_model(x, sigma_curr)
                    drift = (sigma_curr ** 2 - sigma_next ** 2) * score

                    if step < num_steps - 1:
                        z = torch.randn_like(x)
                        diffusion = torch.sqrt(sigma_curr ** 2 - sigma_next ** 2) * z
                    else:
                        diffusion = 0.0

                    x = x + drift + diffusion
                    
                result[i * batch_size:(i + 1) * batch_size] = x.detach().cpu().numpy()
        return result
    
    def sample_with_history(self, batch_size, num_steps=1000, delim=10):
        self.score_model.eval()
        
        t_init = torch.zeros(batch_size, device=self.device)
        sigma_max = self.scheduler.get_sigma(t_init).view(-1, 1)
        x = torch.randn(batch_size, 2).to(self.device) * sigma_max
        history = [x.detach().cpu().numpy()]

        with torch.no_grad():
            for step in range(num_steps):
                t = torch.full((batch_size,), step / num_steps, device=self.device)
                sigma_curr = self.scheduler.get_sigma(t).view(-1, 1)
                sigma_next = self.scheduler.get_sigma(torch.clamp(t + 1.0 / num_steps, max=1.0)).view(-1, 1)

                score = self.score_model(x, sigma_curr)
                drift = (sigma_curr ** 2 - sigma_next ** 2) * score

                if step < num_steps - 1:
                    z = torch.randn_like(x)
                    diffusion = torch.sqrt(sigma_curr ** 2 - sigma_next ** 2) * z
                else:
                    diffusion = 0.0

                x = x + drift + diffusion

                if step % delim == 0:
                    history.append(x.detach().cpu().numpy())
        return np.array(history)


class DDIM_sampler:
    # DDIM Style Sampler
    def __init__(self, score_model, scheduler=schedule_sigma(), eta=0.0,device='cpu'):
        self.score_model = score_model
        self.scheduler = scheduler
        self.device = device
        self.eta = eta
    
    def sample(self, batch_size, num_batches, num_steps=1000):
        self.score_model.eval()
        
        result = np.zeros((num_batches * batch_size, 2), dtype=np.float32)

        t_init = torch.zeros(batch_size, device=self.device)
        sigma_max = self.scheduler.get_sigma(t_init).view(-1, 1)
        
        with torch.no_grad():
            for i in range(num_batches):
                x = torch.randn(batch_size, 2).to(self.device) * sigma_max
                for step in range(num_steps):
                    t = torch.full((batch_size,), step / num_steps, device=self.device)
                    sigma_curr = self.scheduler.get_sigma(t).view(-1, 1)
                    sigma_next = self.scheduler.get_sigma(torch.clamp(t + 1.0 / num_steps, min=0.0)).view(-1, 1)

                    score = self.score_model(x, sigma_curr)
                    pred_noise = -sigma_curr * score
                    x0_pred = x - sigma_curr * pred_noise

                    sigma_rand = self.eta * torch.sqrt(sigma_curr ** 2 - sigma_next ** 2)
                    sigma_det = torch.sqrt(torch.clamp(sigma_next**2 - sigma_rand**2, min=0.0))

                    dir_vector = sigma_det * pred_noise

                    if self.eta > 0:
                        z = torch.randn_like(x) if step < num_steps - 1 else 0.0
                        diffusion = sigma_rand * z
                    else:
                        diffusion = 0.0

                    x = x0_pred + dir_vector + diffusion
                    
                result[i * batch_size:(i + 1) * batch_size] = x.detach().cpu().numpy()
        return result
    
    def sample_with_history(self, batch_size, num_steps=1000, delim=10):
        self.score_model.eval()
        
        t_init = torch.zeros(batch_size, device=self.device)
        sigma_max = self.scheduler.get_sigma(t_init).view(-1, 1)
        x = torch.randn(batch_size, 2).to(self.device) * sigma_max
        history = [x.detach().cpu().numpy()]

        with torch.no_grad():
            for step in range(num_steps):
                t = torch.full((batch_size,), step / num_steps, device=self.device)
                sigma_curr = self.scheduler.get_sigma(t).view(-1, 1)
                sigma_next = self.scheduler.get_sigma(torch.clamp(t + 1.0 / num_steps, min=0.0)).view(-1, 1)

                score = self.score_model(x, sigma_curr)
                pred_noise = -sigma_curr * score
                x0_pred = x - sigma_curr * pred_noise

                sigma_rand = self.eta * torch.sqrt(sigma_curr ** 2 - sigma_next ** 2)
                sigma_det = torch.sqrt(torch.clamp(sigma_next**2 - sigma_rand**2, min=0.0))

                dir_vector = sigma_det * pred_noise

                if self.eta > 0:
                    z = torch.randn_like(x) if step < num_steps - 1 else 0.0
                    diffusion = sigma_rand * z
                else:
                    diffusion = 0.0

                x = x0_pred + dir_vector + diffusion

                if step % delim == 0:
                    history.append(x.detach().cpu().numpy())
        return np.array(history)