import logging
from logging import FileHandler, StreamHandler
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

# --- Logging Setup ---

def setup_logging(log_file: str):
    """
    Configures the root logger to send logs to two destinations:
    1. A log file.
    2. The console.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any default handlers to avoid duplicate output
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Define the format for our logs
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s"
    )

    # 1. Add a file handler to write logs to a file
    # mode='w' will overwrite the log file each time the script starts
    file_handler = FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 2. Add a console handler to print logs to the terminal
    console_handler = StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


# --- Flower Server Code (from your provided script) ---
DEVICE = torch.device("cpu")

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

def get_evaluate_fn(testset):
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        net = Net().to(DEVICE)
        set_parameters(net, parameters)
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
        # Use the logger instead of print to capture the output
        logging.info(f"Server-side evaluation accuracy: {accuracy * 100:.2f}%")
        return loss, {"accuracy": accuracy}
    return evaluate

if __name__ == "__main__":
    # Configure logging to console and file at startup
    setup_logging(log_file="server.log")

    # We need a dataset for the centralized evaluation function
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = CIFAR10("./dataset", train=False, download=True, transform=transforms)

    # 1. Define the strategy for Federated Evaluation
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        fraction_evaluate=1.0,
        min_evaluate_clients=2,
        evaluate_fn=get_evaluate_fn(testset),
    )

    # 2. Start the server
    server_address = "0.0.0.0:8080"
    logging.info(f"Starting Flower server with Federated Evaluation on {server_address}")

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    logging.info("--- Flower server has shut down. ---")