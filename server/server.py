# server.py
import argparse
import logging
from logging import FileHandler, StreamHandler
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
# ---
class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for tabular data."""

    def __init__(self, num_features, num_classes) -> None:
        super(MLP, self).__init__()
        # Changed print to logging.info
        logging.info(f"Initializing server model: {num_features} features, {num_classes} classes")
        print(
            f"Initializing server model: {num_features} features, {num_classes} classes"
        )
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---
# 2. Helper Functions for Parameter Handling (Unchanged)
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
    config: Dict[str, fl.common.Scalar] = {
        "server_round": server_round,  # Pass the current round number
        "local_epochs": 2,             # Tell clients to train for 2 local epochs
    }
    # Changed print to logging.info
    logging.info(f"Server: Sending fit config for round {server_round}: {config}")
    return config


def evaluate_metrics_aggregation(
    all_client_metrics: List[Tuple[int, Dict[str, fl.common.Scalar]]],
) -> Dict[str, fl.common.Scalar]:
    """Aggregate evaluation metrics from all clients."""
    # Changed print to logging.info
    logging.info(f"Server: Aggregating metrics from {len(all_client_metrics)} clients...")
    total_examples = sum([num_examples for num_examples, _ in all_client_metrics])

    if total_examples == 0:
        # Changed print to logging.warning
        logging.warning("Warning: No examples found for metric aggregation.")
        return {"accuracy": 0.0}

    weighted_accuracy = 0.0
    for num_examples, metrics in all_client_metrics:
        if "accuracy" in metrics:
            accuracy = metrics["accuracy"]
            if not isinstance(accuracy, (float, int)):
                # Changed print to logging.warning
                logging.warning(f"Warning: Skipping non-scalar accuracy metric: {accuracy}")
                continue
            weighted_accuracy += num_examples * accuracy
        else:
            # Changed print to logging.warning
            logging.warning(f"Warning: 'accuracy' metric not found in client metrics: {metrics}")

    aggregated_accuracy = weighted_accuracy / total_examples
    # Changed print to logging.info
    logging.info(f"Server: Aggregated weighted accuracy: {aggregated_accuracy:.4f}")
    return {"accuracy": aggregated_accuracy}


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = 0  # Will be set from the config

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[
            Union[
                Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes],
                BaseException,
            ]
        ],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate fit results and save the model on the final round."""

        # 1. Call the parent class's aggregate_fit to perform the aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # 2. Save the model parameters ONLY on the final round
        if aggregated_parameters is not None and server_round == self.num_rounds:
            print(f"Saving final model after round {server_round}...")

            # Convert parameters to a list of numpy arrays
            ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save the numpy arrays to a .npz file in the server's directory
            np.savez("final_model.npz", *ndarrays)
            print("Final model saved to final_model.npz")

        return aggregated_parameters, aggregated_metrics


# ---
# 4. Main Server Execution
# ---
if __name__ == "__main__":
    # STEP 1: Set up the logger to write to a file and the console.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    file_handler = FileHandler("server.log", mode='w')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    
    # --- (Argument parsing is unchanged) ---
    parser = argparse.ArgumentParser(description="Flower Server for CSV Data")

    parser.add_argument(
        "--features",
        type=int,
        required=True,
        help="The total number of input features for the model (run CSVDataset.py to find this).",
    )
    parser.add_argument(
        "--classes",
        type=int,
        required=True,
        help="The total number of output classes for the model (run CSVDataset.py to find this).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Total number of federated learning rounds.",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=1,
        help="Minimum number of clients required to start a round.",
    )

    args = parser.parse_args()

    # Initialize the global model
    net = MLP(num_features=args.features, num_classes=args.classes).to(DEVICE)
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(net))
    logging.info("Server model initialized.")

    # 2. Define the strategy, using our new SaveModelStrategy
    strategy = SaveModelStrategy(  # <-- USE THE CUSTOM STRATEGY
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_fit_config_fn=fit_config,
        fraction_evaluate=1.0,
        min_evaluate_clients=args.clients,
        fraction_fit=1.0,
        min_fit_clients=args.clients,
        min_available_clients=args.clients,
    )
    strategy.num_rounds = args.rounds  # Pass the total number of rounds to the strategy

    # Start the server
    server_address = "0.0.0.0:8080"
    logging.info(f"Starting Flower server at {server_address} for {args.rounds} rounds...")
    logging.info(f"Waiting for at least {args.clients} client(s) to connect...")

    # STEP 2: Wrap the blocking server call in a try...finally block.
    try:
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
    finally:
        # STEP 3: This code ALWAYS runs, guaranteeing the log file is saved.
        logging.shutdown()
        # Use print here because logging is now off.
        print("\nServer shutdown. Log file flushed.")
