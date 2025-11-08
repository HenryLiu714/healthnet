# CSVDataset.py
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main(args):
    """Loads CSV, processes it, and saves it in PyTorch format."""
    print("Starting dataset processing...")
    input_path = Path(args.input)
    out_dir = Path(args.out)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created at: {out_dir}")

    # 1. Load CSV
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise
    print(f"CSV loaded successfully. Shape: {df.shape}")

    # 2. Define features and target based on your CSV image
    # Note: Column names must match your CSV file exactly.
    try:
        target_col = 'Health Risk Level'
        numerical_cols = ['Age', 'BMI', 'Alcohol Consumption', 'Physical Activity', 'Sleep Duration', 'Stress Level']
        categorical_cols = ['Gender', 'Smoking Status', 'Chronic Disease History']

        # Ensure all defined columns exist in the DataFrame
        all_cols = numerical_cols + categorical_cols + [target_col]
        missing_cols = [col for col in all_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: The following columns are missing from the CSV: {missing_cols}")
            raise ValueError("CSV columns do not match expected schema.")

    except KeyError as e:
        print(f"Error: Missing expected column in CSV: {e}")
        raise

    # 3. Preprocessing
    print("Preprocessing data...")

    # --- Handle potential missing values ---
    # For simplicity, we'll fill numerical with median and categorical with mode
    for col in numerical_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            print(f"Warning: Missing values found in '{col}'. Filling with median: {median_val}")
            df[col] = df[col].fillna(median_val)

    for col in categorical_cols + [target_col]:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            print(f"Warning: Missing values found in '{col}'. Filling with mode: {mode_val}")
            df[col] = df[col].fillna(mode_val)

    # --- Process Features (X) ---
    print(f"Scaling numerical features: {numerical_cols}")
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(df[numerical_cols])

    print(f"One-hot encoding categorical features: {categorical_cols}")
    # pd.get_dummies will create k columns for k categories
    X_categorical = pd.get_dummies(df[categorical_cols], drop_first=False)

    # Combine processed features
    X_numerical_df = pd.DataFrame(X_numerical, columns=numerical_cols)
    X_final = pd.concat([X_numerical_df, X_categorical.reset_index(drop=True)], axis=1)

    # --- Process Target (y) ---
    print(f"Factorizing target column: {target_col}")
    # pd.factorize will create 0-indexed integer labels
    y_labels, unique_classes = pd.factorize(df[target_col])

    num_features = X_final.shape[1]
    num_classes = len(unique_classes)

    print(f"Found {num_features} total features after processing.")
    print(f"Found {num_classes} target classes: {list(unique_classes)}")

    # 4. Split data
    print("Splitting data into train and test sets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_final.values, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    # 5. Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 6. Save data as .pt files
    train_data = (X_train_tensor, y_train_tensor)
    test_data = (X_test_tensor, y_test_tensor)

    train_path = out_dir / "train.pt"
    test_path = out_dir / "test.pt"

    torch.save(train_data, train_path)
    torch.save(test_data, test_path)
    print(f"Saved training data to: {train_path}")
    print(f"Saved testing data to: {test_path}")

    # 7. Save metadata for the client
    metadata = {
        "num_features": num_features,
        "num_classes": num_classes,
        "classes_map": list(unique_classes)
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved dataset metadata to: {meta_path}")

    print("\n" + "="*30)
    print("  Dataset Processing Complete!")
    print(f"  > Total Features: {num_features}")
    print(f"  > Total Classes:  {num_classes}")
    print("  Use these values to start your server:")
    print(f"  python server.py --features {num_features} --classes {num_classes}")
    print("="*30 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV for PyTorch-based FL")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file."
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to the output directory to save processed .pt files."
    )

    args = parser.parse_args()
    main(args)