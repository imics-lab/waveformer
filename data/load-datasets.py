import numpy as np
import torch
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_ucr_dataset(dataset_name):
    X_train_raw, y_train_raw = load_classification(dataset_name, split="train")
    X_test_raw,  y_test_raw  = load_classification(dataset_name, split="test")

    # Encode string labels → integer class indices
    le = LabelEncoder()
    le.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train_enc = le.transform(y_train_raw)
    y_test_enc  = le.transform(y_test_raw)

    # Handle unequal length: aeon returns a list instead of ndarray
    def pad_to_array(X_list):
        if isinstance(X_list, np.ndarray):
            # Equal length — already (samples, channels, timesteps)
            return X_list.transpose(0, 2, 1)   # → (samples, timesteps, channels)
        # Unequal length — list of (channels, timesteps) arrays
        max_len = max(x.shape[-1] for x in X_list)
        padded = []
        for x in X_list:
            pad_width = max_len - x.shape[-1]
            # Pad with zeros on the right along the timestep axis
            x_padded = np.pad(x, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            padded.append(x_padded)
        padded = np.stack(padded)                # (samples, channels, timesteps)
        return padded.transpose(0, 2, 1)         # → (samples, timesteps, channels)

    X_train_np = pad_to_array(X_train_raw)
    X_test_np  = pad_to_array(X_test_raw)

    # Split test into valid + test (50/50)
    X_valid_np, X_test_np, y_valid_enc, y_test_enc = train_test_split(
        X_test_np, y_test_enc, test_size=0.5, random_state=42, stratify=y_test_enc
    )

    # Convert to torch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_valid = torch.tensor(X_valid_np, dtype=torch.float32)
    X_test  = torch.tensor(X_test_np,  dtype=torch.float32)
    y_train = torch.tensor(y_train_enc, dtype=torch.int64)
    y_valid = torch.tensor(y_valid_enc, dtype=torch.int64)
    y_test  = torch.tensor(y_test_enc,  dtype=torch.int64)

    print(f"Dataset: {dataset_name}")
    print(f"  Classes: {list(le.classes_)} → {list(range(len(le.classes_)))}")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    t_names = list(le.classes_)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, t_names