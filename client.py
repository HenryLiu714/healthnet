import argparse
from collections import OrderedDict
import warnings

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the same neural network as the server
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define training and testing functions
def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

# Function to load a specific data partition
def load_data(partition_id: int):
    """Load a partition of the CIFAR-10 dataset."""
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transforms)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transforms)

    # Split training set into 2 partitions
    partition_size = len(trainset) // 2
    lengths = [partition_size, len(trainset) - partition_size]
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
    
    # Get the correct partition
    train_partition = datasets[partition_id]
    
    # Create DataLoaders
    trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)
    # The client will use the full test set for validation
    valloader = DataLoader(testset, batch_size=32)
    return trainloader, valloader # Return both loaders

# Define the Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # This is not strictly necessary for FedAvg, but it's good practice
        # and required by the NumPyClient interface.
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader) # valloader is None
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    # 1. Parse command line argument
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument(
        "--partition",
        type=int,
        choices=[0, 1],
        required=True,
        help="Partition of the dataset to use for this client.",
    )
    args = parser.parse_args()

    # 2. Load the model and data
    net = Net().to(DEVICE)
    trainloader, testloader = load_data(args.partition)
    
    print(f"Client {args.partition} loaded {len(trainloader.dataset)} training examples.")

    # 3. Start the Flower client
    client = FlowerClient(net, trainloader, testloader)
    server_address = "127.0.0.1:8080"
    
    print(f"Client {args.partition} connecting to server at {server_address}")
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )