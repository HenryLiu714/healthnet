from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

DEVICE = torch.device("cpu")  # Use CPU for the server


# Define the same neural network as the client
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


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# Define a function for centralized evaluation (optional)
def get_evaluate_fn(testset):
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        net = Net().to(DEVICE)
        set_parameters(net, parameters)  # Update model with the latest parameters
        testloader = DataLoader(testset, batch_size=32)
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        print(f"Server-side evaluation accuracy: {accuracy * 100:.2f}%")
        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    # 1. Define the strategy for Federated Evaluation
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        # We also need to tell the strategy to run evaluation on clients
        fraction_evaluate=1.0,  # Evaluate on all connected clients
        min_evaluate_clients=2,  # Minimum number of clients for evaluation
    )

    # 2. Start the server
    server_address = "0.0.0.0:8080"
    print(f"Starting Flower server with Federated Evaluation on {server_address}")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
