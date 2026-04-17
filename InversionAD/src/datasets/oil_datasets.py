import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os


class OilDataset(Dataset):
    def __init__(self, data_root, img_size=4, train=True, transform_type='none', **kwargs):
        self.train = train
        self.img_size = img_size
        self.window_size = img_size * img_size

        file_name = "train_oil.csv" if train else "test_oil.csv"
        csv_path = os.path.join(data_root, file_name)
        df = pd.read_csv(csv_path)
        returns = df['log_returns'].values

        # Pobieramy flagi filtrowania z kwargs
        normal_only = kwargs.get('normal_only', False)
        anom_only = kwargs.get('anom_only', False)

        if not train and 'is_anomaly' in df.columns:
            raw_labels = df['is_anomaly'].values
        else:
            raw_labels = np.zeros(len(returns))

        self.mean, self.std = returns.mean(), returns.std()
        normalized_data = (returns - self.mean) / (self.std + 1e-8)

        self.samples = []
        self.final_labels = []
        step = self.window_size

        for i in range(0, len(normalized_data) - self.window_size + 1, step):
            window = normalized_data[i: i + self.window_size]
            label = 1 if np.any(raw_labels[i : i + self.window_size] == 1) else 0

            # --- KLUCZOWY DODATEK: FILTROWANIE ---
            if not train:
                if normal_only and label == 1: continue
                if anom_only and label == 0: continue
            # --------------------------------------

            self.samples.append(window.reshape(1, self.img_size, self.img_size))
            self.final_labels.append(label)

        print(f"Załadowano {'treningowy' if train else 'testowy'} zbiór: {len(self.samples)} okien.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            'samples': torch.from_numpy(self.samples[idx]).float(),
            'clslabels': 0,
            'labels': self.final_labels[idx]
        }