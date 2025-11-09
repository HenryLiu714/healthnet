# run_client.py
import argparse
import json
import warnings
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---
# 1. Define Model for Tabular Data (MLP)
# This replaces the CNN from the example.
# ---
class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for tabular data."""
    def __init__(self, num_features, num_classes) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---
# 2. Define Training and Testing
# ---
def train(net, trainloader, epochs, server_round):
    """Train the model on the training set."""
    print(f"Starting training for {epochs} epoch(s)...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(epochs):
        epoch_loss, correct, total = 0.0, 0, 0
        for features, labels in trainloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        # Calculate metrics for this epoch
        final_epoch_loss = epoch_loss / len(trainloader)
        final_epoch_acc = correct / total

        # --- THIS IS THE KEY PART ---
        # Print metrics as a JSON string for the backend to parse
        metrics_payload = {
            "type": "metrics",
            "round": int(server_round),
            "epoch": epoch + 1,
            "loss": final_epoch_loss,
            "accuracy": final_epoch_acc
        }
        print(json.dumps(metrics_payload))
        # ----------------------------

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {final_epoch_loss:.4f} | Train Acc: {final_epoch_acc:.4f}")


def test(net, testloader):
    """Validate the model on the test set."""
    print("Starting evaluation on test set...")
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    avg_loss = loss / len(testloader)
    accuracy = correct / total
    print(f"Evaluation complete. Test Loss: {avg_loss:.4f} | Test Acc: {accuracy:.4f}")
    return avg_loss, accuracy


# ---
# 3. Define Data Loading
# ---
def load_data(data_dir: Path):
    """Load train/test data prepared by CSVDataset.py"""
    print(f"Loading data from {data_dir}...")

    # Load metadata
    try:
        with open(data_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        num_features = metadata["num_features"]
        num_classes = metadata["num_classes"]
        print(f"Loaded metadata: {num_features} features, {num_classes} classes.")
    except FileNotFoundError:
        print(f"Error: metadata.json not found in {data_dir}.")
        raise

    # Load tensor data
    try:
        X_train, y_train = torch.load(data_dir / "train.pt")
        X_test, y_test = torch.load(data_dir / "test.pt")
    except FileNotFoundError:
        print(f"Error: train.pt or test.pt not found in {data_dir}.")
        raise

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valloader = DataLoader(test_dataset, batch_size=32)

    print(f"Loaded {len(train_dataset)} training examples and {len(test_dataset)} test examples.")
    return trainloader, valloader, num_features, num_classes


# ---
# 4. Define Flower Client
# ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        """Get model parameters as a list of NumPy arrays."""
        print("Client: get_parameters called")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        print("Client: set_parameters called")
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model."""
        print("Client: fit called")
        self.set_parameters(parameters)

        # Get config from server
        epochs = config.get("local_epochs", 1)
        server_round = config.get("server_round", 0)

        # Train the model
        train(self.net, self.trainloader, epochs=epochs, server_round=server_round)

        # Return new parameters, number of examples, and metrics
        print("Client: fit complete")
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model."""
        print("Client: evaluate called")
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader)

        # Return loss, number of examples, and metrics
        print("Client: evaluate complete")
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


# ---
# 5. Main execution
# ---
if __name__ == "__main__":
    # 1. Parse command line arguments from backend
    parser = argparse.ArgumentParser(description="Flower client for CSV data")

    # Argument 1: --server (from backend)
    parser.add_argument(
        "--server",
        type=str,
        required=True,
        help="Address of the Flower server"
    )

    # Argument 2: --data (from backend)
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the directory containing processed data"
    )

    args = parser.parse_args()
    print(f"Client script started. Server: {args.server}, Data: {args.data}")

    try:
        # 2. Load the model and data
        data_dir = Path(args.data)
        trainloader, valloader, num_features, num_classes = load_data(data_dir)

        net = MLP(num_features=num_features, num_classes=num_classes).to(DEVICE)
        print(f"Model {type(net).__name__} loaded on {DEVICE}.")

        # 3. Start the Flower client
        client = FlowerClient(net, trainloader, valloader)

        print(f"Connecting to Flower server at {args.server}")
        fl.client.start_numpy_client(
            server_address=args.server,
            client=client,
        )
        print("Flower client connection closed.")

    except Exception as e:
        # Print errors to stdout so the backend can catch them
        print(f"Client script failed: {e}")
        import traceback
        traceback.print_exc()
        # This will be caught by the backend as a log/error
        raise