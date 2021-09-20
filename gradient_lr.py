import torch
import torch.nn as nn
from torch.nn import parameter
from torch.utils import data
import random

def general_data(w, b, n_example):
    X = torch.normal(0, 1, (n_example, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1,1))


def load_data(data_tensor, batch_size):
    dataset = data.TensorDataset(*data_tensor)
    return data.DataLoader(dataset, batch_size, shuffle=True)

def data_iter(feature, labels, batch_size):
    n_example = len(feature)
    indices = list(range(n_example))
    random.shuffle(indices)
    for i in range(0, n_example, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size, n_example)])
        yield feature[batch_indices], labels[batch_indices]

true_w = torch.tensor([2, -3.4])
true_b = 4.2
n_example = 1000
feature, labels = general_data(true_w, true_b, n_example)
model = nn.Sequential(nn.Linear(2,1))

lr = 0.03
num_epoch = 100
lf = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
dataLoader = load_data((feature, labels), batch_size=16)

for epoch in range(num_epoch):
    for i, (X, y) in enumerate(dataLoader):
        optimizer.zero_grad()
        y_hat = model(X)
        loss = lf(y_hat, y)
        loss.backward()
        optimizer.step()
    l = lf(model(X), y)
    print(f'epoch {epoch + 1}, loss {l:f}')
