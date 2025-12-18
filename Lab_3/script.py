import torch
import random
torch.set_printoptions(sci_mode=False)

# 1. Создаём тензор x целочисленного типа, хранящий случайное значение
x = torch.randint(low=1, high=10, size=(1,), dtype=torch.int32)
print("x =", x)

x = x.to(torch.float32)
x.requires_grad = True
print("x преобразованный =", x)

# 2. Устанавливаем степень n 
n = 2   

# 3. Выполняем операции
# 3.1 Возведение в степень n
a = x ** n

# 3.2 Умножение на случайное число от 1 до 10
k = random.randint(1, 10)
b = a * k

# 3.3 Взятие экспоненты
y = torch.exp(b)

print("Случайный множитель k =", k)
print("y =", y)

# 4. Вычисление производной dy/dx
y.backward()

print("dy/dx =", x.grad)
