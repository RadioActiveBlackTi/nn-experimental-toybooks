import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
from sinusoidal import SineWaveTask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class simpleNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=40, output_size=1):
        super(simpleNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
    
    def forward(self, x, params=None):
        if params is None:
            x = self.tanh(self.linear1(x))
            x = self.tanh(self.linear2(x))
            x = self.output(x)
        else:
            x = self.tanh(F.linear(x, params['linear1.weight'], params['linear1.bias']))
            x = self.tanh(F.linear(x, params['linear2.weight'], params['linear2.bias']))
            x = F.linear(x, params['output.weight'], params['output.bias'])
        return x


def train_baseline(model, tasks, epochs=1000, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for task in tasks:
            x_train, y_train = task.sample_data(10)
            x_val, y_val = task.sample_data(10)

            x_train, y_train = x_train.to(device), y_train.to(device)
            x_val, y_val = x_val.to(device), y_val.to(device)

            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                y_val_pred = model(x_val)
                val_loss = loss_fn(y_val_pred, y_val)
                total_loss += val_loss.item()
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}, Validation Loss: {total_loss/len(tasks)}")


def train_maml(model, tasks, meta_epochs=1000, inner_lr=0.01, meta_lr=0.001, k_shot=10, adapt_steps=1):
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    loss_fn = nn.MSELoss()

    for epoch in range(meta_epochs):
        total_loss = 0.0
        for task in tasks:
            x_train, y_train = task.sample_data(k_shot)
            x_val, y_val = task.sample_data(k_shot)

            x_train, y_train = x_train.to(device), y_train.to(device)
            x_val, y_val = x_val.to(device), y_val.to(device)

            # Copy model parameters
            fast_weights = OrderedDict(model.named_parameters())

            # Inner loop adaptation
            for _ in range(adapt_steps):
                pred = model(x_train, params=fast_weights)
                loss = loss_fn(pred, y_train)

                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

                # SGD
                fast_weights = OrderedDict(
                    (name, param - inner_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), grads)
                )

            # Compute validation loss with adapted parameters
            y_val_pred = model(x_val, params=fast_weights)
            val_loss = loss_fn(y_val_pred, y_val)

            total_loss += val_loss

        # Meta-optimization step
        meta_optimizer.zero_grad()
        total_loss.backward()
        meta_optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}, Meta Loss: {total_loss.item()/len(tasks)}")

def train_reptile(model, tasks, meta_epochs=1000, inner_lr=0.01, meta_lr=0.001, k_shot=10, adapt_steps=1):

    loss_fn = nn.MSELoss()

    for epoch in range(meta_epochs):
        total_loss = 0.0
        meta_grad_buffer = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        for task in tasks:
            x_train, y_train = task.sample_data(k_shot)
            x_val, y_val = task.sample_data(k_shot)

            x_train, y_train = x_train.to(device), y_train.to(device)
            x_val, y_val = x_val.to(device), y_val.to(device)

            # Copy model parameters
            fast_weights = OrderedDict(model.named_parameters())

            # Inner loop adaptation
            for _ in range(adapt_steps):
                pred = model(x_train, params=fast_weights)
                loss = loss_fn(pred, y_train)
                
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=False)

                max_norm = 1.0 
                total_norm = 0.0
                for g in grads:
                    param_norm = g.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                clip_coef = max_norm / (total_norm + 1e-6)
                if clip_coef < 1:
                    grads = [g * clip_coef for g in grads]

                grads = [torch.clamp(g, -1.0, 1.0) for g in grads]

                fast_weights = OrderedDict(
                    (name, param - inner_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), grads)
                )

            # Validation Loss
            with torch.no_grad():
                y_val_pred = model(x_val, params=fast_weights)
                val_loss = loss_fn(y_val_pred, y_val)
                total_loss += val_loss

            for name, param in model.named_parameters():
                diff = fast_weights[name].detach() - param.detach()
                meta_grad_buffer[name] += diff
        

        # Meta-optimization step
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data += meta_lr * (meta_grad_buffer[name] / len(tasks))

        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}, Meta Loss: {total_loss.item()/len(tasks)}")


def eval_maml(model, k_shot=10, inner_lr=0.01, adapt_steps=10, model_name="MAML"):
    task = SineWaveTask()
    task.amplitude = 3.0 
    task.phase = 0.0 
    
    x_support, y_support = task.sample_data(k_shot)

    x_support, y_support = x_support.to(device), y_support.to(device)
    
    x_all = torch.linspace(-5, 5, 100).view(-1, 1)
    y_true = task.amplitude * torch.sin(x_all + task.phase)
    
    with torch.no_grad():
        y_pre = model(x_all.to(device), params=None)
    
    # Copy model parameters
    fast_weights = OrderedDict(model.named_parameters())
    
    loss_fn = nn.MSELoss()
    
    for i in range(adapt_steps):
        pred = model(x_support, params=fast_weights)
        loss = loss_fn(pred, y_support)
        
        grads = torch.autograd.grad(loss, fast_weights.values())
        
        # SGD
        fast_weights = OrderedDict(
            (name, param - inner_lr * grad)
            for ((name, param), grad) in zip(fast_weights.items(), grads)
        )

    with torch.no_grad():
        y_post = model(x_all.to(device), params=fast_weights)

    plt.figure(figsize=(10, 6))
    
    plt.plot(x_all.numpy(), y_true.numpy(), label='Ground Truth (Sine)', color='gray', linestyle='--')
    plt.scatter(x_support.cpu().numpy(), y_support.cpu().numpy(), label=f'Support Set ({k_shot} points)', color='red', s=80, marker='x')
    plt.plot(x_all.numpy(), y_pre.cpu().numpy(), label='Pre-update (Initialization)', color='green', alpha=0.5, linestyle=':')
    plt.plot(x_all.numpy(), y_post.cpu().numpy(), label=f'Post-update ({adapt_steps} steps)', color='blue', linewidth=2)

    plt.title(f'{model_name} Adaptation Result (K={k_shot})')
    plt.legend()
    plt.ylim(-4, 4)
    plt.grid(True, alpha=0.3)
    plt.show()
