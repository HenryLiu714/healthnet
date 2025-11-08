from torch.utils.data import Dataset
import pandas as pd

class CSVDataset(Dataset):
    def __init__(self, csv_file, feature_cols, label_col, transform=None):
        """
        csv_file: path to CSV file
        feature_cols: list of column names or indices for input features
        label_col: column name or index for target label
        transform: optional transform for the features
        """
        self.data = pd.read_csv(csv_file)
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get features and label
        features = self.data.iloc[idx][self.feature_cols].values.astype(float)
        label = self.data.iloc[idx][self.label_col]

        # Convert to PyTorch tensors
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)  # use float if regression

        # Apply optional transforms
        if self.transform:
            features = self.transform(features)

        return features, label
