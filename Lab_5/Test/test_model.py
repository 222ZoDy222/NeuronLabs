import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import random

# SETTINGS
MODE = 'folder'      # 'folder' or 'random'
NUM_IMAGES = 10      # used only in random mode

DATA_TEST_PATH = './data/test'
FOR_TEST_PATH = './FOR_TEST'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

class_names = ['beer', 'pizza']

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

net = CnNet(len(class_names)).to(device)
net.load_state_dict(torch.load('CnNet_safe.ckpt', map_location=device))
net.eval()

print('Model loaded')

transform = transforms.Compose([
    transforms.Resize(68),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])


if MODE == 'folder':
    image_list = sorted([
        os.path.join(FOR_TEST_PATH, f)
        for f in os.listdir(FOR_TEST_PATH)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

    if not image_list:
        raise RuntimeError('FOR_TEST folder is empty')

elif MODE == 'random':
    image_list = None

else:
    raise ValueError('MODE must be "folder" or "random"')


def load_random_image():
    cls = random.choice(class_names)
    cls_path = os.path.join(DATA_TEST_PATH, cls)
    img_name = random.choice(os.listdir(cls_path))
    path = os.path.join(cls_path, img_name)
    return Image.open(path).convert('RGB'), path


if MODE == 'folder':

    for path in image_list:

        img = Image.open(path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = net(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        plt.figure()
        plt.imshow(img)
        plt.title(
            f'Prediction: {class_names[pred]} | '
            f'Confidence: {probs[0][pred]:.2f}\n'
            f'File: {os.path.basename(path)}\n'
            f'Press Enter for next image'
        )
        plt.axis('off')
        plt.show(block=False)

        plt.waitforbuttonpress()  
        plt.close()

elif MODE == 'random':

    for _ in range(NUM_IMAGES):

        img, path = load_random_image()
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = net(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        plt.figure()
        plt.imshow(img)
        plt.title(
            f'Prediction: {class_names[pred]} | '
            f'Confidence: {probs[0][pred]:.2f}\n'
            f'File: {os.path.basename(path)}\n'
            f'Press Enter for next image'
        )
        plt.axis('off')
        plt.show(block=False)

        plt.waitforbuttonpress()
        plt.close()

print('Done')
