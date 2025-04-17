import torch
import time
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        # describe the operation
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=(11, 11), stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 8)
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(64 * 62 * 62, num_of_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # define transforms
    train_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
    ])

    model = AlexNet().to(device)

    path = Path("images")

    dataset = torchvision.datasets.ImageFolder(root=path, transform=train_transform)
    dls = DataLoader(dataset, batch_size=64, shuffle=True)
    print(f"number of images: {len(dataset)}")
    print(f"number of batches: {len(dls)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 3
    total_step = len(dls)

    print(torch.dtype)

    t0 = time.monotonic()

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dls):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    t1 = time.monotonic()
    print(f"training durationt: {t1 - t0}\n")
