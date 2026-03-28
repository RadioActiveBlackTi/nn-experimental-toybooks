import torch
import torch.nn as nn
import torch.nn.functional as F

from n_halfmoons import trainloader, trainset
from score_net import GaussianFourierProjection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class score_model_cfg(nn.Module):
    def __init__(self, input_dim=2, embed_dim=128, hidden_dim=128, scale=5.0, output_dim=2, num_classes=6):
        super(score_model_cfg, self).__init__()

        self.embed = GaussianFourierProjection(input_dim=input_dim, embedding_size=embed_dim, scale=scale)

        self.noise_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.class_embed = nn.Embedding(num_classes + 1, embed_dim) # (num_classes) for unconditional class

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, sigma, class_label):
        x_embed = self.embed(x)
        t_embed = self.noise_embed(torch.log(sigma).view(-1, 1))
        class_embed = self.class_embed(class_label)

        x_embed = x_embed + t_embed + class_embed

        out = F.silu(self.fc1(x_embed))
        out = F.silu(self.fc2(out))
        out = self.fc3(out)
        return out


def train_NCSN_cfg(model, scheduler, optimizer, drop_prob=0.1, num_epochs=100):
    model.train()
    losses = []
    num_classes = 6
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in trainloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_size = batch_x.size(0)

            mask = (torch.rand(batch_size, device=device) < drop_prob).long()
            batch_y = batch_y * (1 - mask) + num_classes * mask  # num_classes is the unconditional class

            t = torch.rand(batch_size, device=device)

            sigma = scheduler.get_sigma(t).view(-1, 1)

            z = torch.randn_like(batch_x)
            x_noisy = batch_x + sigma * z

            class_labels = batch_y

            score_pred = model(x_noisy, sigma, class_labels)

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


class CFG_score(nn.Module):
    def __init__(self, score_model, guidance_scale=1.0, guidance_index=0, unconditional_class=6):
        super(CFG_score, self).__init__()
        self.score_model = score_model
        self.guidance_scale = guidance_scale
        self.guidance_index = guidance_index
        self.unconditional_class = unconditional_class
    
    def set_guidance_scale(self, scale):
        self.guidance_scale = scale
    
    def set_guidance_index(self, index):
        self.guidance_index = index
    
    def forward(self, x, sigma):
        batch_size = x.size(0)
        class_labels = torch.full((batch_size,), self.guidance_index, device=x.device, dtype=torch.long)
        unconditional_labels = torch.full((batch_size,), self.unconditional_class, device=x.device, dtype=torch.long)

        score_cond = self.score_model(x, sigma, class_labels)
        score_uncond = self.score_model(x, sigma, unconditional_labels)

        guided_score = score_uncond + self.guidance_scale * (score_cond - score_uncond)
        return guided_score
