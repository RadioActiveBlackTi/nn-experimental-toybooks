import torch
import torch.nn as nn
from torchdiffeq import odeint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, t, x):
        t_vec = torch.ones(x.shape[0], 1).to(x.device) * t
        x_and_t = torch.cat([x, t_vec], dim=1)       
        return self.net(x_and_t)


class FFJORD(nn.Module):
    def __init__(self, odefunc):
        super(FFJORD, self).__init__()
        self.odefunc = odefunc

    def forward(self, t, states):
        # states: [zt, logp_z]
        zt, logp_z = states

        with torch.enable_grad():
            zt.requires_grad_(True)
            dzdt = self.odefunc(t, zt)
            # Compute trace of Jacobian
            epsilon = torch.randn_like(zt)
            grad_z = torch.autograd.grad(dzdt, zt, epsilon, create_graph=True)[0]
            trace = torch.sum(grad_z * epsilon, dim=1)
        dlogp_zdt = -trace.view(-1, 1)
        return dzdt, dlogp_zdt


class ODEModel(nn.Module):
    def __init__(self, odefunc, num_steps=10, dopri=False, rtol=1e-3, atol=1e-3):
        super(ODEModel, self).__init__()
        self.ffjord = FFJORD(odefunc)
        self.num_steps = num_steps
        self.dopri = dopri
        self.rtol = rtol
        self.atol = atol
        
    def forward(self, x):
        batch_size = x.size(0)
        logp_z = torch.zeros(batch_size, 1).to(device)
        if not self.dopri:
            integration_times = torch.linspace(0., 1., steps=self.num_steps).to(device)
            zt, logp_z = odeint(self.ffjord, (x, logp_z), integration_times, method='rk4')
        else:
            zt, logp_z = odeint(self.ffjord, (x, logp_z), torch.tensor([0., 1.]).to(device), rtol=self.rtol, atol=self.atol)

        return zt[-1], logp_z[-1]