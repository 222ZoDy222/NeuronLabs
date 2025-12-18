import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_simple.csv")

X = torch.Tensor(df.iloc[:, 0:2].values)

# нормализуем метки
y_raw = df.iloc[:, 2].astype(str).str.strip().str.lower()

print("Уникальные значения класса:", y_raw.unique())

y = torch.Tensor(np.where(y_raw == "купит", 1, -1).reshape(-1,1))

class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
            nn.Tanh()
        )
    def forward(self, X):
        return self.layers(X)

inputSize = X.shape[1]   
hiddenSize = 3
outputSize = 1

net = NNet(inputSize, hiddenSize, outputSize)

with torch.no_grad():
    pred0 = net(X)

pred0_lbl = torch.Tensor(np.where(pred0 >= 0, 1, -1).reshape(-1, 1))
err0 = torch.sum(torch.abs(y - pred0_lbl)) / 2
print("Ошибок до обучения:", int(err0.item()))

# ====== ОБУЧЕНИЕ ======
lossFn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

epochs = 201
for i in range(1, epochs):
    pred = net(X)
    loss = lossFn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 20 == 0:
        print(f"Ошибка на {i}-й итерации: {loss.item()}")

with torch.no_grad():
    pred = net(X)

pred_lbl = torch.Tensor(np.where(pred >= 0, 1, -1).reshape(-1, 1))
err = torch.sum(torch.abs(y - pred_lbl)) / 2
print("\nОшибок после обучения:", int(err.item()))
