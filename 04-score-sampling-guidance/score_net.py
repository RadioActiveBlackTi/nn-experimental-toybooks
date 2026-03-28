import torch
import torch.nn as nn
import torch.nn.functional as F

from n_halfmoons import trainloader, trainset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainloader.to(device)
trainset.to(device)

# Feature Augmentation with Gaussian Fourier Projections
class GaussianFourierProjection(nn.Module):
    def __init__(self, input_dim=3, embedding_size=256, scale=10.0):
        super(GaussianFourierProjection, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, embedding_size // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x @ self.W
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    

class score_network(nn.Module):
    def __init__(self, input_dim=2, embed_dim=128, hidden_dim=128, scale=5.0, output_dim=2):
        super(score_network, self).__init__()

        self.embed = GaussianFourierProjection(input_dim=input_dim, embedding_size=embed_dim, scale=scale)

        self.noise_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Softplus(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, sigma):
        x_embed = self.embed(x)
        t_embed = self.noise_embed(torch.log(sigma).view(-1, 1))
        x_embed = x_embed + t_embed

        out = F.softplus(self.fc1(x_embed))
        out = F.softplus(self.fc2(out))
        out = self.fc3(out)
        return out

def train_NCSN(model, scheduler, optimizer, num_epochs=100):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, _ in trainloader:
            batch_x = batch_x.to(device)
            batch_size = batch_x.size(0)

            t = torch.rand(batch_size, device=device)

            sigma = scheduler.get_sigma(t).view(-1, 1)

            z = torch.randn_like(batch_x)
            x_noisy = batch_x + sigma * z

            score_pred = model(x_noisy, sigma)

            loss = (((score_pred * sigma + z) ** 2)).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_size
        epoch_loss /= len(trainset)
        losses.append(epoch_loss)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return losses