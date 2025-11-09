# server.py
import argparse
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")  # Server typically runs on CPU

# ---
# 1. Define Model for Tabular Data (MLP)
#    This MUST match the model defined in run_client.py
# ---
class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for tabular data."""
    def __init__(self, num_features, num_classes) -> None:
        super(MLP, self).__init__()
        print(f"Initializing server model: {num_features} features, {num_classes} classes")
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ---
# 2. Helper Functions for Parameter Handling
# ---
def get_parameters(net) -> List[np.ndarray]:
    """Get model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    """Set model parameters from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# ---
# 3. Strategy Configuration Functions
# ---

def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Return training configuration dict for each round.

    This is sent to the client's `fit()` method.
    """
    config = {
        "server_round": server_round,  # Pass the current round number
        "local_epochs": 2,             # Tell clients to train for 2 local epochs
    }
    print(f"Server: Sending fit config for round {server_round}: {config}")
    return config

def evaluate_metrics_aggregation(
    all_client_metrics: List[Tuple[int, Dict[str, fl.common.Scalar]]]
) -> Dict[str, fl.common.Scalar]:
    """Aggregate evaluation metrics from all clients."""
    print(f"Server: Aggregating metrics from {len(all_client_metrics)} clients...")

    # This function receives a list of tuples: (num_examples, metrics_dict)
    # We will average the 'accuracy' metric, weighted by the number of examples

    total_examples = sum([num_examples for num_examples, _ in all_client_metrics])

    if total_examples == 0:
        print("Warning: No examples found for metric aggregation.")
        return {"accuracy": 0.0} # Avoid division by zero

    weighted_accuracy = 0.0
    for num_examples, metrics in all_client_metrics:
        if "accuracy" in metrics:
            accuracy = metrics["accuracy"]
            if not isinstance(accuracy, (float, int)):
                print(f"Warning: Skipping non-scalar accuracy metric: {accuracy}")
                continue
            weighted_accuracy += num_examples * accuracy
        else:
            print(f"Warning: 'accuracy' metric not found in client metrics: {metrics}")

    aggregated_accuracy = weighted_accuracy / total_examples

    print(f"Server: Aggregated weighted accuracy: {aggregated_accuracy:.4f}")

    # Return the aggregated metric
    return {"accuracy": aggregated_accuracy}

# ---
# 4. Main Server Execution
# ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Server for CSV Data")

    parser.add_argument(
        "--features",
        type=int,
        required=True,
        help="The total number of input features for the model (run CSVDataset.py to find this)."
    )
    parser.add_argument(
        "--classes",
        type=int,
        required=True,
        help="The total number of output classes for the model (run CSVDataset.py to find this)."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Total number of federated learning rounds."
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=1,
        help="Minimum number of clients required to start a round."
    )

    args = parser.parse_args()

    # 1. Initialize the global model
    net = MLP(num_features=args.features, num_classes=args.classes).to(DEVICE)
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(net))
    print("Server model initialized.")

    # 2. Define the strategy
    # We use FedAvg and configure it to:
    # - Use our metric aggregation function
    # - Use our fit config function
    # - Wait for at least `args.clients` clients
    strategy = fl.server.strategy.FedAvg(
        initial_parameters=initial_parameters,

        # --- Federated Evaluation ---
        # Ask clients to evaluate and aggregate their metrics
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        fraction_evaluate=1.0,  # Evaluate on all connected clients
        min_evaluate_clients=args.clients,

        # --- Federated Training ---
        on_fit_config_fn=fit_config,  # Send config (e.g., epochs) to clients
        fraction_fit=1.0,             # Train on all connected clients
        min_fit_clients=args.clients,

        min_available_clients=args.clients,
    )

    # 3. Start the server
    server_address = "0.0.0.0:8080"
    print(f"Starting Flower server at {server_address} for {args.rounds} rounds...")
    print(f"Waiting for at least {args.clients} client(s) to connect...")

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    print("Server shutdown.")