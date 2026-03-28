from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# Load the Cora dataset with feature normalization
# 'root' specifies where to store the data
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

# Print information about the dataset
print(f'Dataset: {dataset.name}')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Graph data object: {data}')