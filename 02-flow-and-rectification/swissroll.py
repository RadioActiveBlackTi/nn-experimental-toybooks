from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = make_swiss_roll(n_samples=5000, noise=0.3, random_state=42)
dataset = StandardScaler().fit_transform(dataset[0])
data = torch.tensor(dataset[:, [0, 2]], dtype=torch.float32).to(device)

train_loader = torch.utils.data.DataLoader(dataset = data,
                                           batch_size = 256,
                                           shuffle = True)