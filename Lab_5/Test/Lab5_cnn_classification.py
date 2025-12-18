import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

class SafeImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.samples = self._filter_valid_samples(self.samples)

    def _filter_valid_samples(self, samples):
        valid_samples = []
        print('Проверка изображений')
        for path, label in samples:
            try:
                with Image.open(path) as img:
                    img.convert('RGB')
                valid_samples.append((path, label))
            except Exception:
                print(f'Удалён битый файл: {path}')
        print(f'Валидных изображений: {len(valid_samples)}')
        return valid_samples

    def loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

data_transforms = transforms.Compose([
    transforms.Resize(68),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])

train_dataset = SafeImageFolder(
    root='./data/train',
    transform=data_transforms
)

test_dataset = SafeImageFolder(
    root='./data/test',
    transform=data_transforms
)

class_names = train_dataset.classes
num_classes = len(class_names)

print('Классы:', class_names)
print('Train size:', len(train_dataset))
print('Test size:', len(test_dataset))


batch_size = 10

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

class CnNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(8 * 8 * 64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

net = CnNet(num_classes).to(device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

num_epochs = 200
loss_history = []

start = time.time()
for epoch in range(num_epochs):
    net.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}], Loss: {loss.item():.4f}')

print('Training time:', time.time() - start)

plt.figure()
plt.plot(loss_history)
plt.title('Training loss')
plt.show()

net.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')

torch.save(net.state_dict(), 'CnNet_safe.ckpt')

inputs, _ = next(iter(test_loader))
inputs = inputs.to(device)
outputs = net(inputs)
_, preds = torch.max(outputs, 1)

for img, cls in zip(inputs.cpu(), preds.cpu()):
    img = img.numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(class_names[cls])
    plt.axis('off')
    plt.pause(1)
    plt.show()
