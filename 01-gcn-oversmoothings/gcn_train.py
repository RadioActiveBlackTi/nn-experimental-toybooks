import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, data, optimizer, criterion, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

def evaluate(model, data):
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask].to(device)).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc

if __name__ == "__main__":
    import BSGCN
    from analysis import plot_gcn_oversmoothing_analysis
    from cora_load import data
    model = BSGCN.BSGCN(in_channels=data.num_features, hidden_channels=16, out_channels=data.num_classes, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    train(model, data, optimizer, criterion, epochs=200)
    acc = evaluate(model, data)
    print(f'Test Accuracy: {acc:.4f}')

    # --- Analysis ---
    model.eval()
    _, activations_dict = model.forward_inspect(data.x.to(device), data.edge_index.to(device))
    plot_gcn_oversmoothing_analysis(activations_dict, data.edge_index, num_nodes=data.num_nodes)