import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

def make_halfmoons(n_samples_per_class=1000, n_classes=6, noise=0.2, outer_radius=10.0, inner_radius=5.0):

    X = np.zeros((n_samples_per_class * n_classes, 2), dtype=np.float32)
    y = np.zeros(n_samples_per_class * n_classes, dtype=np.int64)
    for j in range(n_classes):
        ix = range(n_samples_per_class * j, n_samples_per_class * (j + 1))
        a = outer_radius
        phi = j * 2 * np.pi / n_classes
        r = inner_radius + noise * np.random.randn(n_samples_per_class)
        t = np.linspace(0, np.pi, n_samples_per_class)
        X[ix] = np.c_[a * np.sin(phi) + r * np.sin(t + phi), a * np.cos(phi) + r * np.cos(t + phi)]
        y[ix] = j
    return X, y

X, y = make_halfmoons(n_samples_per_class=2000, n_classes=6, noise=0.4, inner_radius=4.0, outer_radius=7.0)
X = StandardScaler().fit_transform(X)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
trainset = torch.utils.data.TensorDataset(torch.tensor(train_X), torch.tensor(train_y))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torch.utils.data.TensorDataset(torch.tensor(test_X), torch.tensor(test_y))
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)