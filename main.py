import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as data
import matplotlib.pyplot as plt
import time

class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        # warstwy konwolucyjne
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # warstwy w pełni połączone
        self.fc1 = nn.Linear(16 * 61 * 61, 120)  # Aktualizacja wymiarów w warstwach Fully Connected
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        # przepływ danych przez sieć neuronową
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 61 * 61)  # Aktualizacja wymiarów
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Definicja transformacji dla danych treningowych i testowych
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Zmiana rozmiaru na 256x256
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Ścieżki do katalogów 'train' i 'test' z podkatalogami klas
train_dir = 'train'
test_dir = 'test'

# Wczytanie danych treningowych i testowych z uwzględnieniem podkatalogów
train_data = ImageFolder(train_dir, transform=train_transforms)
test_data = ImageFolder(test_dir, transform=train_transforms)

# Tworzenie DataLoaderów
trainloader = data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = data.DataLoader(test_data, batch_size=64, shuffle=False)

# Inicjalizacja modelu, funkcji straty i optymalizatora
net = DeepNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Trenowanie modelu przez 50 epok
for epoch in range(50):
    running_loss = 0.0
    start_time = time.time()

    for i, data in enumerate(trainloader, 0):
        net.train()
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 7 == 6:
            # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 7))
            running_loss = 0.0

    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Pomiar czasu trwania epoki
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f'Epoch {epoch + 1} took {epoch_time:.2f} seconds')
    print('Testing accuracy: %.3f %%' % (100 * correct / total))