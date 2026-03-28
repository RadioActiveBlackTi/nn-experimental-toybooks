import torch
import torch.nn as nn
import torch.nn.functional as F

from n_halfmoons import trainloader, testloader, trainset, testset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Classifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_classes=6):
        super(Classifier, self).__init__()

        self.x_fc = nn.Linear(input_dim, hidden_dim)
        self.sigma_fc = nn.Linear(1, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, sigma):
        x = F.relu(self.x_fc(x))
        sigma = F.relu(self.sigma_fc(torch.log(sigma).view(-1, 1)))
        x_embed = x + sigma

        out = self.fc(x_embed)
        return out

def train_classifier(model, optimizer, scheduler, num_epochs=100):
    model.train()
    train_losses = []
    test_losses = []
    accuracies = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in trainloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_size = batch_x.size(0)

            t = torch.rand(batch_size, device=device)

            sigma = scheduler.get_sigma(t).view(-1, 1)

            logits = model(batch_x, sigma)

            loss = F.cross_entropy(logits, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_size
        epoch_loss /= len(trainset)
        train_losses.append(epoch_loss)

        accuracy = 0.0
        for batch_x, batch_y in testloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_size = batch_x.size(0)

            t = torch.rand(batch_size, device=device)

            sigma = scheduler.get_sigma(t).view(-1, 1)

            logits = model(batch_x, sigma)

            loss = F.cross_entropy(logits, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_size

            accuracy += (logits.argmax(dim=1) == batch_y).float().sum().item()
        epoch_loss /= len(testset)
        accuracies.append(accuracy / len(testset))
        test_losses.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Accuracy: {accuracies[-1]:.4f}")


class Guided_score(nn.Module):
    def __init__(self, score_model, classifier, guidance_scale=1.0, guidance_index=0):
        super(Guided_score, self).__init__()
        self.score_model = score_model
        self.classifier = classifier
        self.guidance_scale = guidance_scale
        self.guidance_index = guidance_index
    
    def set_guidance_scale(self, scale):
        self.guidance_scale = scale
    
    def set_guidance_index(self, index):
        self.guidance_index = index

    def forward(self, x, sigma):
        score = self.score_model(x, sigma)

        with torch.enable_grad():
            x.requires_grad_(True)
            logits = self.classifier(x, sigma)
            log_probs = F.log_softmax(logits, dim=1)
            selected_log_probs = log_probs[:, self.guidance_index]
            classifier_loss = selected_log_probs.sum()
            grads = torch.autograd.grad(classifier_loss, x)[0]
        grads = grads.detach() / (torch.norm(grads, dim=-1, keepdim=True) + 1e-8)  # Prevent Gradient Explosion
        guided_score = score + self.guidance_scale * grads
        return guided_score