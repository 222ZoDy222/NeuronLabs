import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==============================
# 1. Загрузка данных
# ==============================
data = pd.read_csv("data.csv")

# 4 признака + столбец 'species'
X = data.iloc[:, :4].values
y = data.iloc[:, 4].values

# Преобразуем классы в числа
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# В тензоры
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Разбиваем на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 2. Модель
# ==============================
# Линейная классификация: вход 4 признака, выход 3 класса
model = nn.Linear(4, 3)

# ==============================
# 3. Функция потерь и оптимизатор
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ==============================
# 4. Обучение
# ==============================
epochs = 1000

for epoch in range(epochs):

    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss = {loss.item():.4f}")

# ==============================
# 5. Оценка точности
# ==============================
with torch.no_grad():
    test_pred = model(X_test)
    predicted_classes = torch.argmax(test_pred, dim=1)

    correct = (predicted_classes == y_test).sum().item()
    total = y_test.size(0)
    accuracy = correct / total

print("Правильных предсказаний:", correct)
print("Всего объектов:", total)
print("Точность:", accuracy)


print("\nТочность на тестовых данных:", accuracy)
